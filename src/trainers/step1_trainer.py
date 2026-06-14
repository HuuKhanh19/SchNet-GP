"""
Step 1 Trainer: SchNet Baseline with Adam.

Standard gradient descent training for molecular property prediction.
Supports regression (RMSE) and classification (AUC).
"""

import os
import time
import json
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
from sklearn.metrics import roc_auc_score


class Step1Trainer:
    """Adam trainer for SchNet baseline."""

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: torch.device,
        experiment_dir: str,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.experiment_dir = experiment_dir
        # Cờ lưu output ra đĩa (checkpoint + results.json). Nếu tắt, best model
        # vẫn được giữ trong RAM để khôi phục lúc test.
        self.save = config.get('experiment', {}).get('save', True)
        if self.save:
            os.makedirs(experiment_dir, exist_ok=True)
        self.best_state = None

        self.task_type = config['dataset']['task_type']
        self.metric_name = config['dataset'].get('metric', 'rmse')

        # Loss
        if self.task_type == 'classification':
            self.criterion = nn.BCELoss()
        else:
            self.criterion = nn.MSELoss()

        # Optimizer
        tcfg = config['training']
        self.optimizer = Adam(
            model.parameters(),
            lr=tcfg.get('learning_rate', 5e-4),
            weight_decay=tcfg.get('weight_decay', 1e-5),
        )

        # Scheduler - mode='min' works for both:
        #   regression: val_score = RMSE (lower is better)
        #   classification: val_score = -AUC (lower is better)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=tcfg.get('scheduler_factor', 0.5),
            patience=tcfg.get('scheduler_patience', 25),
        )

        self.epochs = tcfg.get('epochs', 300)
        self.patience = tcfg.get('early_stopping_patience', 100)
        self.gradient_clip = tcfg.get('gradient_clip', 1.0)

        # Tracking
        self.best_val_metric = float('inf')
        self.best_epoch = 0
        self.no_improve_count = 0
        self.history = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        n_samples = 0

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            target = batch.pop('target')

            self.optimizer.zero_grad()
            output = self.model(batch)
            pred = output['prediction']
            loss = self.criterion(pred, target)
            loss.backward()

            if self.gradient_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )

            self.optimizer.step()
            total_loss += loss.item() * target.shape[0]
            n_samples += target.shape[0]

        return {'loss': total_loss / max(n_samples, 1)}

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_preds = []
        all_targets = []
        total_loss = 0.0
        n_samples = 0

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            target = batch.pop('target')

            output = self.model(batch)
            pred = output['prediction']
            loss = self.criterion(pred, target)

            total_loss += loss.item() * target.shape[0]
            n_samples += target.shape[0]
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())

        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)

        metrics = {'loss': total_loss / max(n_samples, 1)}

        if self.task_type == 'regression':
            metrics['rmse'] = float(np.sqrt(np.mean((preds - targets) ** 2)))
        else:
            try:
                metrics['auc'] = float(roc_auc_score(targets, preds))
            except ValueError:
                metrics['auc'] = 0.0

        return metrics

    @torch.no_grad()
    def predict(self, loader: DataLoader):
        """Trả về (preds, targets) dạng np.ndarray theo đúng thứ tự dataset.

        Dùng cho residual/delta-learning: cần prediction thô để cộng lại baseline
        và tính diagnostic. Loader phải shuffle=False để thứ tự khớp dataset.smiles.
        """
        self.model.eval()
        all_preds, all_targets = [], []
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            target = batch.pop('target')
            pred = self.model(batch)['prediction']
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())
        return np.concatenate(all_preds), np.concatenate(all_targets)

    def get_val_score(self, metrics: Dict[str, float]) -> float:
        """Get validation score (lower is better for early stopping).
        
        Regression: returns RMSE directly.
        Classification: returns -AUC so lower = better AUC.
        """
        if self.task_type == 'classification':
            return -metrics.get('auc', 0.0)
        return metrics.get('rmse', metrics['loss'])

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        print(f"Model: {self.model.num_params:,} params | "
              f"Optimizer: Adam | epochs={self.epochs} | "
              f"batch_size={self.config['training'].get('batch_size', 32)}")

        # Initial evaluation (before any training)
        init_metrics = self.evaluate(valid_loader)
        if self.task_type == 'regression':
            print(f"Initial  | val_rmse={init_metrics['rmse']:.4f}")
        else:
            print(f"Initial  | val_auc={init_metrics.get('auc', 0):.4f}")
        print("-" * 70)

        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(valid_loader)
            val_score = self.get_val_score(val_metrics)

            # Scheduler step
            self.scheduler.step(val_score)

            # Early stopping check
            if val_score < self.best_val_metric:
                self.best_val_metric = val_score
                self.best_epoch = epoch
                self.no_improve_count = 0
                # Giữ best model trong RAM (tránh ghi đĩa mỗi epoch cải thiện).
                self.best_state = copy.deepcopy({
                    k: v.detach().cpu() for k, v in self.model.state_dict().items()
                })
            else:
                self.no_improve_count += 1

            # Log
            elapsed = time.time() - epoch_start
            record = {
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'elapsed': elapsed,
                'lr': self.optimizer.param_groups[0]['lr'],
            }
            if self.task_type == 'regression':
                record['val_rmse'] = val_metrics['rmse']
            else:
                record['val_auc'] = val_metrics.get('auc', 0.0)
            self.history.append(record)

            # Print progress every 5 epochs
            if epoch % 5 == 0 or epoch == 1:
                lr = self.optimizer.param_groups[0]['lr']
                if self.task_type == 'regression':
                    print(
                        f"Epoch {epoch:3d} | "
                        f"train_loss={train_metrics['loss']:.4f} | "
                        f"val_rmse={val_metrics['rmse']:.4f} | "
                        f"best_val={self.best_val_metric:.4f}@ep{self.best_epoch} | "
                        f"lr={lr:.2e} | "
                        f"{elapsed:.1f}s"
                    )
                else:
                    print(
                        f"Epoch {epoch:3d} | "
                        f"train_loss={train_metrics['loss']:.4f} | "
                        f"val_auc={val_metrics.get('auc', 0):.4f} | "
                        f"best_val={-self.best_val_metric:.4f}@ep{self.best_epoch} | "
                        f"lr={lr:.2e} | "
                        f"{elapsed:.1f}s"
                    )

            if self.no_improve_count >= self.patience:
                print(f"\nEarly stopping at epoch {epoch} (best={self.best_epoch})")
                break

        total_time = time.time() - start_time
        print(f"\nTraining done in {total_time:.1f}s ({total_time/60:.1f}min)")

        # Restore best model (từ RAM) rồi đánh giá trên test
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        # Best validation score (regression: best val RMSE; classification: best AUC)
        if self.task_type == 'regression':
            print(f"\nBest val RMSE: {self.best_val_metric:.4f} "
                  f"(epoch {self.best_epoch})")
        else:
            print(f"\nBest val AUC: {-self.best_val_metric:.4f} "
                  f"(epoch {self.best_epoch})")

        test_metrics = {}
        if test_loader is not None:
            test_metrics = self.evaluate(test_loader)
            print(f"Test (best epoch {self.best_epoch}):")
            if self.task_type == 'regression':
                print(f"  RMSE: {test_metrics['rmse']:.4f}")
            else:
                print(f"  AUC:  {test_metrics.get('auc', 0):.4f}")

        results = {
            'step': 1,
            'optimizer': 'Adam',
            'best_epoch': self.best_epoch,
            'total_time_s': total_time,
            'test_metrics': test_metrics,
            'val_best_score': self.best_val_metric,
            'config': self.config,
            'history': self.history,
        }

        # Lưu ra đĩa chỉ khi bật --save
        if self.save:
            if self.best_state is not None:
                torch.save(self.best_state,
                           os.path.join(self.experiment_dir, 'best_model.pt'))
            with open(os.path.join(self.experiment_dir, 'results.json'), 'w') as f:
                json.dump(results, f, indent=2, default=str)

        return results
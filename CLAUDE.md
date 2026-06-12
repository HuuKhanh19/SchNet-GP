# SchNet-GP — quy ước làm việc cho Claude Code

## Nơi chạy
- Máy này (Mac, arm64) KHÔNG có GPU. Chỉ sửa code ở đây, không bao giờ chạy training ở đây.
  - Có env conda `schnet_cpu` (Python 3.10, CPU-only) để smoke/e2e test nhanh. PyG ghim
    `<2.6` + `torch_cluster` (PyG 2.8 đòi pyg-lib, không có wheel mac-arm64).
    Chạy: `PYTHONPATH=$(pwd) /Users/huukhanh/miniconda3/bin/conda run -n schnet_cpu python ...`
- Training chạy trên server `bkai_host` (Windows, conda env `conan_es`, Python 3.10).
  - GPU: 2× NVIDIA RTX 5070 Ti (16 GB mỗi card), driver 576.88, CUDA 12.9, PyTorch 2.8.0+cu129.
  - WDDM driver model. Chọn card qua `gpu=<0|1>` trong config (card 1 hay bận, card 0 thường rảnh).
  - Lưu ý reproducibility: seed thôi KHÔNG đủ trên GPU vì scatter-add/atomic của message
    passing là non-deterministic. Cờ `--deterministic` (mặc định TẮT) bật
    `torch.use_deterministic_algorithms` để các lần chạy cùng seed ra giống nhau (chậm hơn chút).
- Nhánh làm việc: `khanh`. Mọi thay đổi nằm trên `khanh`, không commit vào `main`.

## Đồng bộ
- Code chia sẻ qua GitHub (origin). Để đưa code lên server: commit + push, rồi server pull.
- KHÔNG commit file nặng: checkpoint, log, dataset đều bị gitignore.

## Quy trình một thí nghiệm
1. Sửa code -> `git add -A && git commit -m "<msg>" && git push`
2. Chạy trên server (người dùng làm trong terminal server, hoặc nhờ Claude Code qua ssh):
   `git pull` rồi chạy bằng python của env conan_es, ghi log ra `logs/<ten>.log`.
3. Kéo log về Mac bằng scp để phân tích.

## Cách chạy (argparse, không còn Hydra)
- Config qua `argparse` trong `scripts/run_step1.py` (registry dataset + lắp config ở `src/config.py`). Xem `python scripts/run_step1.py -h` để biết hết hyper + note.
- SchNet gốc = mặc định (K=1 conformer, cutoff=10). Quét 5 split seed + in RMSE
  trung bình ± std chỉ bằng MỘT lệnh (truyền nhiều seed cho `--seed-split`):
  ```
  python scripts/run_step1.py --dataset esol --seed-split 0 1 2 3 4
  ```
- Default: KHÔNG lưu output và KHÔNG deterministic. Cờ hữu ích: `--gpu <0|1|-1>`, `--save` (ghi checkpoint + results.json vào `experiments/`), `--deterministic` (lặp lại được, chậm hơn), `--num-conformers K`, `--cutoff`.
- Cache split + conformer ở `data/processed/<ds>/<split_method>/seed_<seed>/...` (đã key theo split_method).

## GP head (SchNet freeze 1-conf -> DEAP multi-tree GP)
- Code ở `src/gp/` (`features.py` = extract+cache, `gp_head.py` = DEAP GP, `descriptors.py`
  = RDKit 2D/3D). Runner: `scripts/run_gp.py` (xem `-h`). Cần `deap` (đã thêm vào requirements;
  env `schnet_cpu` đã cài; server `conan_es` cần `pip install deap`).
- Một lệnh/seed: train encoder SchNet K=1 (freeze) -> extract conf-emb (mean-pool atom->conf,
  128) + desc2d/desc3d/energy trên K conformer (sort theo energy = routing) -> standardize theo
  train -> DEAP GP head. Quét nhiều seed in mean ± std, so mốc baseline 0.8994 ± 0.0946.
  ```
  python scripts/run_gp.py --dataset esol --seed-split 0 1 2 3 4 --gpu 0
  ```
- Feature cache key theo (ds, split, seed, K) ở `.../seed_<seed>/gp_K<K>/features.pkl`. Lần đầu
  train encoder + extract (cần GPU); tinh chỉnh hyper GP sau đó DÙNG LẠI cache (bỏ qua encoder),
  chỉ truyền `--force-extract` khi muốn extract lại.
- Hyper GP chính (default): `-K 10`, `--num-emb 8 --num-desc3d 2` (q=10), `--d 16`, `--num-2d 8`,
  `--pop 1000` (quần thể khởi tạo) `--mu 1000 --lam 1000` ((μ+λ)), `--generations 200`,
  `--warmup 0` (>0 = w gen đầu chỉ tiến hóa L1). Energy/standardize/denormalize đều seed được;
  routing dùng thứ tự energy đã sort sẵn lúc extract.

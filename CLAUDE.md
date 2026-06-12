# SchNet-GP — quy ước làm việc cho Claude Code

## Nơi chạy
- Máy này (Windows 11 Pro, x64) CÓ GPU: 1× NVIDIA RTX 5070 Ti (16 GB), driver 591.86.
  Vai trò: sửa code + smoke/e2e test nhanh tại chỗ. Training nặng vẫn đẩy lên server.
  - conda ở `C:\Users\ADMIN\miniconda3`. CẦN TẠO env `conan_es` (Python 3.10, torch CUDA build
    cu12x + PyG) để chạy — hiện máy mới chỉ có `base`, `subsetsel`, `tspwin`, chưa có env cho project.
    PyG ghim `torch_geometric<2.6` + `torch_cluster` (PyG 2.8 đòi pyg-lib, không có wheel cho
    cấu hình này) -> radius_graph fallback qua torch_cluster.
    Chạy (PowerShell): `$env:PYTHONPATH=$PWD; conda run -n conan_es python scripts\run_step1.py ...`
  - Máy này chỉ có 1 card -> dùng `--gpu 0` (hoặc `--gpu -1` để ép CPU).
- Training nặng chạy trên server `bkai_host` (Windows, conda env `conan_es`, Python 3.10).
  - GPU: 2× NVIDIA RTX 5070 Ti (16 GB mỗi card), driver 576.88, CUDA 12.9, PyTorch 2.8.0+cu129.
  - WDDM driver model. Chọn card qua `--gpu <0|1>` (card 1 hay bận, card 0 thường rảnh).
- Lưu ý reproducibility (cả local lẫn server, vì cả hai đều có GPU): seed thôi KHÔNG đủ vì
  scatter-add/atomic của message passing là non-deterministic. Cờ `--deterministic` (mặc định TẮT)
  bật `torch.use_deterministic_algorithms` để các lần chạy cùng seed ra giống nhau (chậm hơn chút).
- Nhánh làm việc: `khanh`. Mọi thay đổi nằm trên `khanh`, không commit vào `main`.
  (Thư mục hiện CHƯA phải git repo — cần `git init` + thêm remote `origin` trước khi dùng quy trình dưới.)
- KHI ĐƯỢC YÊU CẦU PUSH: push lên nhánh `3ai` (không phải `khanh`). Người dùng vào server
  `git pull` nhánh `3ai` rồi chạy. Chỉ push khi người dùng nói rõ "push code lên".

## Đồng bộ
- Code chia sẻ qua GitHub (origin). Để đưa code lên server: commit + push, rồi server pull.
- KHÔNG commit file nặng: checkpoint, log, dataset đều bị gitignore.

## Quy trình một thí nghiệm
1. Sửa code -> `git add -A && git commit -m "<msg>" && git push`
2. Chạy trên server (người dùng làm trong terminal server, hoặc nhờ Claude Code qua ssh):
   `git pull` rồi chạy bằng python của env conan_es, ghi log ra `logs/<ten>.log`.
3. Kéo log về máy local bằng scp để phân tích.

## Cách chạy (argparse, không còn Hydra)
- Config qua `argparse` trong `scripts/run_step1.py` (registry dataset + lắp config ở `src/config.py`). Xem `python scripts/run_step1.py -h` để biết hết hyper + note.
- SchNet gốc = mặc định (K=1 conformer, cutoff=10). Quét 5 split seed + in RMSE
  trung bình ± std chỉ bằng MỘT lệnh (truyền nhiều seed cho `--seed-split`):
  ```
  python scripts/run_step1.py --dataset esol --seed-split 0 1 2 3 4
  ```
- Default: KHÔNG lưu output và KHÔNG deterministic. Cờ hữu ích: `--gpu <0|1|-1>`, `--save` (ghi checkpoint + results.json vào `experiments/`), `--deterministic` (lặp lại được, chậm hơn), `--num-conformers K`, `--cutoff`.
- Cache split + conformer ở `data/processed/<ds>/<split_method>/seed_<seed>/...` (đã key theo split_method).

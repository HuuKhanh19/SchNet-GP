# SchNet-GP — quy ước làm việc cho Claude Code

## Nơi chạy
- Máy này (Mac) KHÔNG có GPU. Chỉ sửa code ở đây, không bao giờ chạy training ở đây.
- Training chạy trên server `bkai_host` (Windows, conda env conan_es).
- Nhánh làm việc: `khanh`. Mọi thay đổi nằm trên `khanh`, không commit vào `main`.

## Đồng bộ
- Code chia sẻ qua GitHub (origin). Để đưa code lên server: commit + push, rồi server pull.
- KHÔNG commit file nặng: checkpoint, log, dataset đều bị gitignore.

## Quy trình một thí nghiệm
1. Sửa code -> `git add -A && git commit -m "<msg>" && git push`
2. Chạy trên server (người dùng làm trong terminal server, hoặc nhờ Claude Code qua ssh):
   `git pull` rồi chạy bằng python của env conan_es, ghi log ra `logs/<ten>.log`.
3. Kéo log về Mac bằng scp để phân tích.

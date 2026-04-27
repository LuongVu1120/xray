# Thư mục ảnh & video cho README / portfolio

Bạn **chỉ cần copy file** vào đúng chỗ bên dưới, rồi `git add`, `commit`, `push`. Không cần sửa code.

## Cấu trúc

```
docs/media/
├── README.md          ← file này
├── images/            ← ảnh PNG, JPG, WebP, GIF
│   ├── hero.png       ← (tuỳ chọn) ảnh bìa / banner
│   ├── ui-agent.png   ← (tuỳ chọn) tab Agent
│   └── ui-classifier.png
└── videos/            ← video MP4, WebM (nên < 50–100 MB để Git nhẹ)
    └── demo.mp4       ← (tuỳ chọn) walkthrough
```

### Gợi ý tên file

| File | Nội dung gợi ý |
|------|----------------|
| `images/hero.png` | Một ảnh tổng quan (UI + kết quả) làm thumbnail repo |
| `images/ui-agent.png` | Màn hình chạy AI Agent, có heatmap + báo cáo |
| `images/ui-classifier.png` | Màn Classifier đơn giản |
| `videos/demo.mp4` | Quay màn hình: upload ảnh → chạy agent → xem báo cáo |

Bạn có thể đặt tên khác; khi đó sửa đường dẫn trong `README.md` gốc của repo cho khớp.

## Sau khi thêm file

Trong thư mục gốc `xray-ai`:

```bash
git add docs/media/
git commit -m "docs: add demo images and video"
git push origin main
```

## Nhúng vào README gốc

Mở `README.md` ở root, trong mục **Demo ảnh & video** đã có sẵn ví dụ. Bỏ comment hoặc thay tên file cho đúng file bạn vừa thêm.

Ví dụ ảnh:

```markdown
![AI Agent](docs/media/images/ui-agent.png)
```

Ví dụ video (GitHub thường hiển thị tốt với đường dẫn tương đối khi đã push):

```markdown
https://github.com/USER/REPO/raw/main/docs/media/videos/demo.mp4
```

Hoặc chỉ đặt link tải:

```markdown
[Xem video hướng dẫn](docs/media/videos/demo.mp4)
```

## Ảnh / video quá lớn

- GitHub khuyến nghị repo gọn; file video nặng nên đăng lên **GitHub Releases**, **YouTube** (unlisted), hoặc **Google Drive** rồi dán link vào README.
- Hoặc dùng [Git LFS](https://git-lfs.github.com/) cho file media lớn.

## Lưu ý pháp lý (y tế)

Chỉ đưa vào repo ảnh **công khai** (ví dụ NIH), **đã ẩn danh**, hoặc **chụp màn hình ứng dụng** không lộ dữ liệu bệnh nhân thật.

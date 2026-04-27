# Media — ảnh & video demo

File đã được **đặt tên gọn** (không khoảng trắng) để GitHub và Markdown ổn định.

## Ảnh (`images/`)

| File | Gợi ý nội dung |
|------|----------------|
| `demo-hero.png` | Ảnh tổng quan / nổi bật UI |
| `demo-step-1.png` → `demo-step-4.png` | Chuỗi màn hình theo thứ tự thao tác |

## Video (`videos/`)

| File | Ghi chú |
|------|---------|
| `demo-walkthrough.mp4` | Quay màn hình toàn bộ luồng (~40 MB) |

Nếu repo chậm khi clone, có thể chuyển video lên **YouTube (unlisted)** hoặc **GitHub Releases** rồi chỉ giữ link trong `README.md` gốc.

## Thêm / thay file mới

Giữ quy tắc: **chữ thường, dấu gạch ngang**, ví dụ `demo-step-5.png`. Sau đó:

```bash
git add docs/media
git commit -m "docs: update demo media"
git push
```

Và cập nhật phần **Demo** trong `README.md` ở root nếu đổi tên file.

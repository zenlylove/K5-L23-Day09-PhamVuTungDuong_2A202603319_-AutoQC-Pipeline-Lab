# Báo cáo Day 09 — Phạm Vũ Tùng Dương, 2A202603319

## 1. Kết luận trước
**Hold.** Intake quét 224 ảnh, loại 31 và chuyển 193 ảnh sang QC; dataset nháp xuất ra có 193 ảnh. Sau mô phỏng duyệt 80 ảnh không còn lỗi bắt buộc, nhưng còn cảnh báo chưa duyệt ở 34/193 ảnh (17,6%), cao hơn ngưỡng 10%.

## 2. Lỗi và phân việc
Số dưới đây là số issue ở QC vòng 1, trước autofix (193 ảnh; xem `02_error_list_v1.csv`). Intake đã loại 31/224 ảnh: 3 ảnh nhỏ, 2 ảnh hỏng, 8 trùng byte, 7 trùng gần và 11 rò rỉ test/val.

| Mã lỗi | Số issue | Phân việc | Lý do |
|---|---:|---|---|
| BAD_CLASS_ID 6; BAD_CLASS_NAME 2; BAD_FORMAT 3; BOX_ZERO 4; DUP_BOX 7 | 22 | Máy | Lỗi cú pháp/ID và box trùng được drop; tên lớp được chuẩn hóa. |
| BOX_OUT_OF_IMAGE | 8 | Máy | Clip box vào biên; nếu clip thành box rỗng thì drop. |
| MISSING_LABEL_FILE | 6 | Người | Không thể tự bịa nhãn; đẩy ảnh lên đầu hàng. |
| BOX_HUGE 3; BOX_TINY 6; EMPTY_LABEL 11; POSSIBLE_FALSE_POS 24; POSSIBLE_LOOSE_BOX 27; POSSIBLE_MISSING 53; POSSIBLE_WRONG_CLASS 17 | 141 | Người xem | Cảnh báo ngữ nghĩa/kích thước, không nên tự sửa theo dự đoán model. |
| CLASS_RARE | 4 | Người phân tích | Cảnh báo cấp dataset, không gắn với một ảnh để xếp vào queue. |

Sau autofix còn 6 lỗi bắt buộc và queue có 121 ảnh, vượt ngân sách 80. Ảnh đầu queue `img_0020` có ưu tiên 11.524 do đồng thời có cảnh báo thiếu/khác lớp và một dòng bị drop; các ảnh thiếu nhãn được ưu tiên 10. Trong 80 ảnh mô phỏng đã duyệt, 74 ảnh có ít nhất một cảnh báo second-opinion. Tổng cảnh báo là nhiều issue hơn số ảnh vì một ảnh có thể có nhiều box/cờ.

## 3. Calibration
| Tham số | Khởi đầu | Đã chọn | Precision / Recall trước → sau | Lý do |
|---|---:|---:|---|---|
| `duplicates.phash_dist` | 2 | 16 | DUP_IMAGE_NEAR: 1.00/0.50 → 1.00/0.83; DUP_IMAGE_EXACT: 1.00/1.00 → 1.00/1.00 | Tăng recall trùng gần mà precision trên calib vẫn 1.00; tránh 20 vì phân loại exact/near và recall thay đổi bất lợi. |
| `second_opinion.conf` | 0.25 | 0.6 | WRONG_CLASS: 0.56/0.83 → 0.62/0.83; MISSING: 0.50/1.00 → 0.57/1.00; LOOSE: 0.29/0.40 → 0.20/0.20 | Chọn 0.6 để tăng precision mà giữ recall của missing/wrong-class; loose-box vẫn là điểm yếu, nên chỉ dùng như cảnh báo cho người. |

Calib chỉ có 39 ảnh và 6 lỗi trùng gần: ở ngưỡng 16, thêm một ca được bắt so với ngưỡng 14 đã làm recall tăng từ 0.67 lên 0.83. Mỗi ca tương đương 16,7 điểm phần trăm, nên không thể coi đây là ước lượng ổn định cho raw. Với ngưỡng đã chọn, intake raw gắn cờ 7 cặp trùng gần; cần mở ảnh kiểm chứng các cặp trước khi coi việc loại ảnh là chính xác.

## 4. Cửa chặn và điểm mù
Chạy `python -m pipeline.run --budget 0 --reports work/rep_nb --export-dir work/exp_nb` cho kết luận **Rework**, mã thoát **20**, còn 6 lỗi bắt buộc và không tạo `work/exp_nb/clean_v1`. Đây là phép thử gate khi không có người xử lý các nhãn thiếu. Trong lần chạy chính, mô phỏng duyệt đã xử lý lỗi bắt buộc nhưng không xem đủ cảnh báo nên quyết định Hold.

Điểm mù có bằng chứng: sau mô phỏng vẫn có 18 cảnh báo `POSSIBLE_LOOSE_BOX`, trong đó một số IoU chỉ nhỉnh hơn ngưỡng `iou_missing`; và 3 `POSSIBLE_WRONG_CLASS` còn lại. Model checker chỉ đưa ý kiến, không có đáp án vàng cho raw để xác nhận nhãn nào đúng. Ngoài ra, nếu model và nhãn cùng bỏ sót hoặc cùng nhầm một vật, so sánh hai bên có thể im lặng; cần kiểm tra thủ công ảnh/nhãn.

## 5. Train: raw vs clean
**Chưa chạy.** Phần train Colab là tùy chọn và chưa có kết quả; không kết luận clean tốt hơn raw khi chưa có các seed và metric so sánh.

## 6. Quyết định và ghi chú AI
Giữ quyết định **Hold**, không phát hành bản nháp. Để xét Release, người cần xử lý/đánh giá cảnh báo còn lại cho tỷ lệ dưới 10% (với 193 ảnh, tối đa 19 ảnh còn cảnh báo) và đảm bảo không còn lỗi bắt buộc.

- **Yêu cầu AI:** hoàn thiện bốn module theo docstring/test trong repo; không sửa dữ liệu hoặc test.
- **Kiểm chứng:** chạy 50 test module, toàn bộ 79 test và pipeline e2e; kiểm tra thêm gate bằng `--budget 0`.
- **Một lỗi AI gặp:** lần đầu second-opinion so IoU với box bị bỏ nhầm một tọa độ; test phát hiện `IndexError`, đã sửa và chạy lại toàn bộ suite thành công.

# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 2026-09-11

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** Python 3.x, PyTorch, Ultralytics YOLO11

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không có thay đổi đáng kể so với notebook gốc; chỉ thực hiện theo hướng dẫn và lưu báo cáo bằng chứng trong file này.

> ZIP do notebook tạo có tên `KX-DAY01-report.zip`. Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`): đây là prediction của model cho cả bức ảnh, không phải nhãn của một đối tượng riêng lẻ. Hạng 1 nghĩa là lớp này có điểm số cao nhất trong danh sách các lớp mà model có thể dự đoán.
- Record này mô tả toàn ảnh như thế nào? Nó mô tả lớp phù hợp nhất với nội dung tổng thể của bức ảnh. Nói cách khác, model đang trả lời câu hỏi: “Ảnh này thuộc lớp nào nhất?”
- Ai định nghĩa class list mà checkpoint có thể dự đoán? Class list do taxonomy và bộ dữ liệu huấn luyện của model quyết định, trong bài lab là các lớp của ImageNet-1K mà checkpoint được huấn luyện.
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy? Vì mỗi trường phục vụ mục đích khác nhau: `class_id` cho máy tính xử lý dữ liệu, `class_name` cho người xem dễ đọc, còn `taxonomy_name` giúp xác định hệ thống phân loại gốc để tránh nhầm lẫn giữa các chuẩn danh mục khác nhau.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì? Guideline cần quy định rõ xem bài toán là single-label hay multi-label, nếu có nhiều vật thể trong cùng ảnh thì có nên gán một nhãn chính hay nhiều nhãn cùng lúc, và cách xử lý nhất quán trong dataset.
- Vì sao model score không phải ground truth? Vì score là xác suất hoặc mức độ tin cậy của model, còn ground truth là nhãn do con người xác nhận theo guideline. Score có thể sai hoặc không chắc chắn, nên nó không thể dùng để thay thế cho nhãn chuẩn.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`): ví dụ một object được model phát hiện với lớp tương ứng, điểm tin cậy và box giới hạn xác định vị trí của vật thể trong ảnh.
- Diễn giải vị trí box bằng lời: box được xác định bằng tọa độ góc trên trái và góc dưới phải, tính theo pixel, với gốc tọa độ ở góc trên trái của ảnh. Tức là box mô tả vùng hình chữ nhật chứa vật thể cần phát hiện.
- So sánh số prediction ở hai threshold: với threshold thấp, model giữ nhiều prediction hơn, tăng độ bao phủ nhưng cũng nhiều false positive; với threshold cao, ít prediction hơn, giảm số sai nhưng có thể bỏ sót object thật.
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem? Threshold thấp làm tăng độ bao phủ nhưng reviewer phải xem thêm nhiều box hơn; threshold cao làm giảm khối lượng review nhưng có nguy cơ mất object thật.
- Đề xuất một quy tắc box chặt: box nên bao kín phần vật thể đã nhìn thấy, đủ lớn để chứa toàn bộ object nhưng không rộng quá mức và không chứa quá nhiều nền không liên quan. Mục tiêu là tối thiểu hóa vùng nền trong box mà vẫn đủ để bao phần đối tượng.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định? Cần có quy định rõ cách xử lý vật thể bị che hoặc cắt mép, ví dụ có đánh dấu ngay phần nhìn thấy hoặc không. Nếu không chắc chắn, cần escalate cho reviewer để đưa ra quyết định thống nhất.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`): mỗi instance có một ID riêng, cùng lớp và một chuỗi điểm xác định biên của đối tượng đó trong không gian 2D.
- Polygon bổ sung chi tiết gì so với box? Box chỉ cho biết vùng bao quanh, còn polygon mô tả hình dạng thực của vật thể. Điều này rất quan trọng khi đối tượng có biên cong, răng cưa, hoặc nhiều góc cạnh.
- `instance_id` dùng để làm gì và không phải loại ID nào? `instance_id` dùng để phân biệt từng instance riêng lẻ trong cùng một ảnh, đặc biệt khi hai object cùng lớp xuất hiện. Nó không phải `class_id`, không phải tracking ID, và không phải ID của người dùng.
- Đề xuất một quy tắc biên mask: mask nên đi sát biên thật của vật thể, không chứa nền và không kéo quá xa ngoài phần đối tượng nhìn thấy. Nếu biên mờ hoặc không chắc chắn, nên ưu tiên thống nhất theo guideline dữ liệu.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định? Cần quy định rõ cách xử lý object chồng lên nhau, mờ, hoặc bị che. Nếu không chắc chắn về biên mask, cần escalate cho reviewer để quyết định thống nhất thay vì gán nhầm.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn lớp cho toàn ảnh theo taxonomy | Ảnh có nhiều chủ thể nhưng không rõ nhãn chính; nhầm tên lớp với taxonomy | Đọc guideline và chọn lớp phù hợp nhất | Kiểm tra nhãn phù hợp với taxonomy và không mơ hồ |
| Phát hiện vật thể | Một instance có class và box `[x_min, y_min, x_max, y_max]` | Box quá rộng/quá hẹp, chồng lấn, mất object, cắt mép | Vẽ box sát vật thể và nhất quán theo guideline | Kiểm tra độ bao phủ, số lượng box, duplicate và label |
| Instance segmentation | Một instance có class và polygon/mask | Biên mờ, object tiếp xúc, mask quá rộng, vùng che khuất | Vẽ polygon sát biên vật thể, tách instance rõ ràng | Xem biên mask, phân tách instance và quyết định vùng mơ hồ |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: Chỉ làm việc với dữ liệu trong phạm vi lab, không lưu, chia sẻ hoặc công khai ảnh, metadata, họ tên, MSSV, email, số điện thoại hoặc thông tin nhạy cảm khác ngoài nội dung bắt buộc.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho mentor/giảng viên hoặc người phụ trách lab để được hướng dẫn xử lý.

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.

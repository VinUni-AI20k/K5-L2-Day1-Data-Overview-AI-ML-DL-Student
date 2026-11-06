# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 6/11/2026

**Runtime Colab:** GPU T4

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=… ; `class_name`=… ; `rank`=1 ; `score`=… ; `taxonomy_name`=…
  {
    "sample_id": "traffic",
    "coco_image_id": 210273,
    "image_width": 640,
    "image_height": 428,
    "task": "image_classification",
    "taxonomy_name": "ImageNet-1K",
    "model_file": "yolo11n-cls.pt",
    "model_sha256": "c62d41bf9625777760018bf914d2e6cd472420ccd01706d97a61cb6c82502bd7",
    "ultralytics_version": "8.4.145",
    "rank": 1,
    "class_id": 468,
    "class_name": "cab",
    "score": 0.510915
  },
3. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
   nói về cả bức ảnh
"task": "image_classification" cho biết đây là tác vụ phân loại ảnh (Image Classification).
"class_name": "cab" là nhãn dự đoán cho nội dung của toàn bộ bức ảnh.
"score": 0.510915 là độ tin cậy của dự đoán lớp cab (taxi).
Không có thông tin tọa độ bounding box (x, y, width, height) để xác định vị trí của một vật thể riêng biệt.
"image_width": 640 và "image_height": 428 mô tả kích thước toàn bộ ảnh.
5. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
Danh sách tên lớp (class list) không phải do model tự nghĩ ra, mà được xác định bởi taxonomy và đi kèm với model/checkpoint.
Trong record trên:
- "taxonomy_name": "ImageNet-1K": cho biết hệ thống phân loại sử dụng bộ nhãn ImageNet-1K.
- "model_file": "yolo11n-cls.pt": checkpoint của mô hình YOLO11 Classification.
- "class_id": 468: ID của lớp được dự đoán.
- "class_name": "cab": tên lớp tương ứng với ID 468.
  
7. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
class_id: Mã số định danh của lớp, giúp máy tính xử lý và đối chiếu lớp một cách nhất quán. Ví dụ: 468.
class_name: Tên dễ đọc của lớp, giúp con người hiểu kết quả dự đoán. Ví dụ: cab.
taxonomy_name: Cho biết hệ thống phân loại hoặc bộ nhãn đang được sử dụng. Ví dụ: ImageNet-1K.

Tránh nhầm lẫn khi các taxonomy khác nhau có thể sử dụng cùng tên lớp nhưng mang ý nghĩa hoặc ID khác nhau.
Giúp kiểm tra và đối chiếu kết quả dự đoán với bộ nhãn gốc.
Hỗ trợ lưu trữ, phân tích và tái sử dụng dữ liệu.
Giúp kết quả dễ hiểu đối với cả con người và máy tính.
9. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
Chọn chủ thể chính: Ưu tiên vật thể hoặc chủ đề nổi bật nhất trong ảnh.
Dựa vào mức độ chiếm diện tích: Xem xét vật thể chiếm phần lớn diện tích ảnh.
Dựa vào ngữ cảnh: Chọn đối tượng thể hiện nội dung hoặc mục đích chính của bức ảnh.
Quy định cách xử lý khi có nhiều đối tượng ngang nhau: Ví dụ, ưu tiên đối tượng ở trung tâm hoặc đối tượng quan trọng nhất theo ngữ cảnh.
Quy định trường hợp không xác định được chủ thể chính: Có thể sử dụng lớp other, ambiguous hoặc chuyển sang người kiểm duyệt.

10. Vì sao không được lấy `score` của model làm ground truth?
Model có thể dự đoán sai: Dù score cao, mô hình vẫn có thể phân loại nhầm đối tượng.
Score không phải nhãn thực tế: Ground truth phải được xác định từ dữ liệu và quá trình gán nhãn đáng tin cậy, không phải từ dự đoán của model.
Score phụ thuộc vào mô hình: Các mô hình khác nhau có thể tạo ra score khác nhau cho cùng một hình ảnh.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…
    {
    "sample_id": "traffic",
    "coco_image_id": 210273,
    "image_width": 640,
    "image_height": 428,
    "task": "object_detection",
    "taxonomy_name": "COCO-80",
    "model_file": "yolo11n.pt",
    "model_sha256": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
    "ultralytics_version": "8.4.145",
    "score_threshold": 0.35,
    "class_id": 5,
    "class_name": "bus",
    "score": 0.912558,
    "coordinate_unit": "pixel",
    "bbox_format": "xyxy",
    "bbox_xyxy": [
      93.17,
      187.95,
      223.01,
      320.91
    ],
    "bbox_width": 129.84,
    "bbox_height": 132.96
  },

3. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
Góc trên-trái: nằm tại tọa độ X = 93.17, Y = 187.95 pixel.
Góc dưới-phải: nằm tại tọa độ X = 223.01, Y = 320.91 pixel.
Chiều rộng: 129.84 pixel.
Chiều cao: 132.96 pixel.

5. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
Với prediction bạn cung cấp có score = 0.912558:
Ngưỡng	Số prediction giữ lại
0.20	1
0.60	1
Vì score 0.912558 lớn hơn cả hai ngưỡng nên prediction được giữ lại ở cả hai trường hợp.

7. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
Kiểm tra các bounding box mới xuất hiện.
Phát hiện các prediction sai (false positives).
Kiểm tra các hộp bị trùng lặp.
Xác minh class name có đúng với đối tượng trong ảnh không.
Quyết định giữ lại, chỉnh sửa hoặc loại bỏ prediction không phù hợp.

9. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
Người gán nhãn phải vẽ bounding box hình chữ nhật bao sát toàn bộ phần nhìn thấy của đối tượng, không bao gồm quá nhiều nền xung quanh. Các cạnh hộp phải được đặt nhất quán theo vị trí thực tế của đối tượng. Đối với đối tượng bị che khuất, chỉ sử dụng phần nhìn thấy và tuân thủ quy định chung về xử lý vật thể bị che khuất

11. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
Tôi sẽ đưa trường hợp này lên hỏi Lab Coach (escalation).

Lý do:
- Không nhìn thấy toàn bộ cơ thể người.
- Chưa rõ guideline yêu cầu gán nhãn phần đối tượng bị cắt mép như thế nào.
- Nếu tự quyết định, mỗi người gán nhãn có thể xử lý khác nhau.
- Cần thống nhất cách xử lý để đảm bảo tính nhất quán của dataset.
- 
## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…
  {
    "sample_id": "traffic",
    "coco_image_id": 210273,
    "image_width": 640,
    "image_height": 428,
    "task": "instance_segmentation",
    "taxonomy_name": "COCO-80",
    "model_file": "yolo11n-seg.pt",
    "model_sha256": "55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152",
    "ultralytics_version": "8.4.145",
    "score_threshold": 0.35,
    "instance_id": "traffic-001",
    "class_id": 5,
    "class_name": "bus",
    "score": 0.925745,
    "coordinate_unit": "pixel",
    "bbox_format": "xyxy",
    "bbox_xyxy": [
      95.3,
      188.72,
      224.08,
      319.96
    ],
    "polygon_point_count": 120,
    "polygon_xy": [
      [
        148.0,
        189.0
      ],
      [
        147.0,
        190.0
      ],
      [
        145.0,
        190.0
      ],
      [
        143.0,
        192.0
      ],
      [
        142.0,
        192.0
      ],
  
3. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
- Bounding box: Chỉ xác định vùng hình chữ nhật bao quanh đối tượng, gồm tọa độ góc trên-trái và góc dưới-phải.
- Polygon: Mô tả đường biên và hình dạng thực tế của đối tượng bằng nhiều điểm tọa độ (x, y).
Trong JSON:
- polygon_point_count: 120 nghĩa là polygon có 120 điểm.
- polygon_xy chứa tọa độ từng điểm trên đường biên của đối tượng.
- bbox_xyxy chỉ mô tả hình chữ nhật bao quanh đối tượng.

5. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
  instance_id dùng để định danh duy nhất một đối tượng cụ thể trong dữ liệu segmentation.
Ý nghĩa:
- instance_id: Mã định danh của một cá thể đối tượng cụ thể.
- class_id: Mã số của loại đối tượng.
- class_name: Tên loại đối tượng.
Class ID xác định đối tượng thuộc loại nào.
Instance ID xác định đó là cá thể nào trong dữ liệu.
Tracking ID dùng để theo dõi cùng một đối tượng qua nhiều khung hình video hoặc nhiều thời điểm.

7. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
Người gán nhãn phải vẽ polygon bám sát đường viền thực tế của phần đối tượng nhìn thấy được. Không vẽ lấn sang nền hoặc các đối tượng khác. Các điểm polygon cần nối liên tục, tạo thành vùng khép kín và hạn chế các điểm thừa không cần thiết. Với vùng bị che khuất, chỉ vẽ phần nhìn thấy theo quy định của guideline.

9. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?

Ví dụ: Những chiếc chảo và nồi treo trên tường ở khu vực phía trên ảnh có một số vị trí chồng lấn hoặc nằm sát nhau, khiến đường biên giữa hai vật không rõ ràng.
Tôi chọn phương án escalation (hỏi Lab Coach).
Lý do:
- Khó xác định chính xác đường biên giữa hai vật.
- Nếu mỗi người tự xử lý, polygon có thể khác nhau.
- Cần thống nhất cách xử lý các vùng tiếp xúc hoặc bị che khuất.
- Lab Coach có thể đưa ra quy tắc chung để áp dụng cho toàn bộ dataset.
## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |Một nhãn lớp đại diện cho toàn bộ ảnh (ví dụ: kitchen).  | JSON dự đoán cab, score 0.510915, nhưng ảnh thực tế là một căn bếp. Có dấu hiệu dự đoán sai lớp. | Chọn một lớp chính theo nội dung toàn ảnh và guideline. | Kiểm tra nhãn có đúng nội dung ảnh không, có nhất quán với taxonomy không. |
| Phát hiện vật thể |  Một hoặc nhiều bounding box (tọa độ xyxy) kèm class ID cho từng vật thể.|  JSON dự đoán bus, score 0.912558, bbox [93.17, 187.95, 223.01, 320.91]. Tuy nhiên, ảnh kitchen không thấy xe bus.| Vẽ box bao sát vật thể, chọn đúng class và xử lý vật bị che hoặc cắt mép theo guideline. | Kiểm tra vị trí box, kích thước, class, vật thể bị bỏ sót và prediction sai. |
| Instance segmentation | Mỗi cá thể có một class và một mask/polygon riêng. | JSON dự đoán bus, score 0.925745, có 120 điểm polygon. Ảnh kitchen có nhiều nồi, chảo chồng lấn nên đường biên giữa các vật khó xác định. | Vẽ polygon theo đường biên phần vật thể nhìn thấy, tách riêng từng cá thể. | Kiểm tra mask có bám sát vật thể không, có lấn sang nền hoặc vật khác không, có bỏ sót vùng không. |



## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS

# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026-11-06

**Runtime Colab:** CPU / Local Python 3.14

**Python / PyTorch / Ultralytics:** Python 3.14.3 / PyTorch 2.15.0+cpu / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không

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

1. **Chép record hạng 1:**
   - `class_id` = `468`
   - `class_name` = `cab`
   - `rank` = `1`
   - `score` = `0.510915`
   - `taxonomy_name` = `ImageNet-1K`

2. **Record này nói về cả bức ảnh hay về một vật thể riêng trong ảnh? Vì sao bạn nghĩ vậy?**
   - Record này nói về **cả bức ảnh** (toàn cảnh).
   - **Vì sao:** Đây là tác vụ Phân loại ảnh (Image Classification). Mô hình tính toán phân phối xác suất trên toàn thể khung hình và đưa ra nhãn tổng thể kèm độ tin cậy. Record hoàn toàn không chứa tọa độ không gian (không có bounding box `bbox_xyxy` hay polygon `polygon_xy`) để khoanh vùng một vật thể riêng lẻ nào. Hơn nữa trường `task` trong dữ liệu JSON được định nghĩa rõ ràng là `"image_classification"`.

3. **Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?**
   - Danh sách tên lớp **đi kèm taxonomy và checkpoint** (ở đây là bộ taxonomy chuẩn `ImageNet-1K` gồm 1.000 lớp cố định được tích hợp sẵn trong checkpoint `yolo11n-cls.pt`).
   - Mô hình học máy/Deep Learning không tự phát minh hay "nghĩ ra" tên lớp mới; mô hình chỉ tính toán trọng số xác suất cho các lớp đã được định nghĩa sẵn trong từ điển nhãn (`names`) của checkpoint.

4. **Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?**
   - `class_id`: Là mã số định danh chuẩn hóa (index), giúp các hệ thống phần mềm, cơ sở dữ liệu và pipeline xử lý chính xác, tránh các lỗi do sai khác chuỗi ký tự, dấu cách hoặc chữ hoa/thường.
   - `class_name`: Giúp con người (annotator, reviewer, kỹ sư AI) đọc hiểu trực quan nội dung ngữ nghĩa.
   - `taxonomy_name`: Xác định không gian nhãn ngữ nghĩa tham chiếu (ví dụ `ImageNet-1K` khác với `COCO-80` hay một ontology tùy chỉnh). Nếu chỉ ghi mỗi tên lớp (như "cab"), hệ thống sẽ không thể biết lớp này được định nghĩa theo quy chuẩn nào, tương ứng với mã ID nào, dẫn đến xung đột hoặc sai lệch khi ánh xạ và tích hợp dữ liệu.

5. **Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn một lớp?**
   - Guideline cần quy định rõ ràng:
     - **Quy tắc chủ thể chính (Dominant object):** Ưu tiên đối tượng có kích thước lớn nhất, chiếm diện tích trung tâm hoặc thu hút sự chú ý rõ rệt nhất của khung hình.
     - **Quy tắc tiền cảnh (Foreground over background):** Ưu tiên vật thể nằm ở tiền cảnh rõ nét hơn là các vật thể mờ phía hậu cảnh.
     - **Thứ tự ưu tiên nghiệp vụ (Taxonomy priority hierarchy):** Khi có nhiều đối tượng cùng xuất hiện (ví dụ vừa có taxi, vừa có xe buýt), guideline cần chỉ rõ loại phương tiện nào được ưu tiên chọn trước theo mục tiêu bài toán.
     - **Quy tắc phân loại tổng thể cảnh quan (Scene vs Object):** Quy định rõ ảnh chụp giao thông đô thị nên được gán theo phương tiện nổi bật hay theo bối cảnh đường phố nếu có lớp tương ứng.

6. **Vì sao không được lấy `score` của model làm ground truth?**
   - `score` (model confidence score) chỉ là xác suất thống kê mô hình ước tính dựa trên các trọng số đã học từ tập dữ liệu huấn luyện trong quá khứ.
   - Mô hình hoàn toàn có thể tự tin cao nhưng dự đoán sai (overconfident mistake) hoặc gặp ảo giác khi gặp bối cảnh lạ, góc chụp phức tạp.
   - `ground truth` là nhãn sự thật khách quan do con người xác thực cẩn thận dựa theo guideline chuẩn. Lấy `score` của model làm ground truth sẽ trực tiếp sao chép các sai số và thiên kiến (bias) của model vào dữ liệu chuẩn, gây suy giảm nghiêm trọng chất lượng của các chu trình huấn luyện tiếp theo.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. **Chép một record:**
   - `class_name` = `person`
   - `score` = `0.912624`
   - `bbox_xyxy` = `[385.33, 69.24, 498.92, 348.92]`
   - `bbox_width` = `113.58`
   - `bbox_height` = `279.68`

2. **Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?**
   - Trong hệ tọa độ ảnh (gốc tọa độ `(0, 0)` tại góc trên cùng bên trái của ảnh):
     - **Góc trên-trái (top-left):** Tọa độ tại `x = 385.33` px (cách mép trái 385.33 pixel) và `y = 69.24` px (cách mép trên đỉnh 69.24 pixel).
     - **Góc dưới-phải (bottom-right):** Tọa độ tại `x = 498.92` px và `y = 348.92` px.
     - **Kích thước hộp:** Hộp giới hạn có chiều rộng `bbox_width = 113.58` pixel (`498.92 - 385.33`) và chiều cao `bbox_height = 279.68` pixel (`348.92 - 69.24`).

3. **Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?**
   - Tại ngưỡng `threshold = 0.20`: Có **17** prediction (phát hiện nhiều đối tượng như person, bowl, oven, cup, spoon, potted plant, dining table, bottle).
   - Tại ngưỡng `threshold = 0.60`: Có **6** prediction (chỉ giữ lại các vật thể có độ tin cậy cao gồm person, bowl, oven).
   *(Ghi chú: Tại ngưỡng chuẩn 0.35 có 11 prediction).*

4. **Khi hạ ngưỡng (giữ nhiều hộp hơn), người reviewer phải làm thêm việc gì?**
   - Khi hạ ngưỡng xuống thấp (giữ thêm nhiều hộp dự đoán yếu), tỷ lệ dương tính giả (false positive) tăng cao.
   - Reviewer phải:
     - Tốn nhiều thời gian rà soát, kiểm tra từng hộp dự đoán có độ tin cậy thấp.
     - Xóa bỏ (reject/delete) các hộp nhận diện sai (như nhận nhầm bóng đổ, vết nứt, mảng màu nền thành đồ vật).
     - Tinh chỉnh lại các hộp bị lệch biên hoặc giải quyết tình trạng nhiều hộp chồng lấn lên cùng một vật thể do ngưỡng NMS không loại trừ hết.

5. **Viết một quy tắc box để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này):**
   - **Quy tắc viền hộp ôm sát (Tight Bounding Box Rule):**  
     > *"Mỗi hộp giới hạn (Bounding box) phải bao trọn toàn bộ các phần nhìn thấy được của vật thể (bao gồm các chi tiết nhô ra như tay cầm, chân, phụ kiện), mép hộp tiếp xúc sát với biên ngoài vật thể với khoảng hở không quá 2-3 pixel; tuyệt đối không cắt lẹm vào thân vật thể và không để thừa khoảng trống nền quá mức."*

6. **Trên ảnh `kitchen`, chỉ ra một vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?**
   - **Vật thể quan sát:** Đối tượng `person` ở mép bên trái ảnh (tọa độ `[0.12, 263.18, 61.55, 310.78]`) chỉ nhìn thấy một phần cánh tay và thân, phần còn lại bị cắt cụt bởi mép ảnh bên trái.
   - **Lựa chọn:** Tôi sẽ **viết thành quy tắc trong guideline (Guideline Rule)**.
   - **Vì sao:** Hiện tượng vật thể bị cắt mép ảnh (truncated) hoặc bị vật khác che khuất một phần (occluded) là tình huống rất thường xuyên và điển hình trong dữ liệu thị giác máy tính thực tế. Việc đưa vào guideline quy tắc rõ ràng (ví dụ: *"Nếu vật thể bị cắt mép nhưng phần nhìn thấy được > 20% và nhận diện rõ ràng thì vẫn đóng box bao trọn phần nhìn thấy được; không ngoại suy phần ngoài ảnh"*) sẽ chuẩn hóa thao tác cho toàn bộ annotator, tránh việc phải liên tục gửi câu hỏi escalation làm nghẽn tiến độ dự án. Escalation chỉ dành cho các trường hợp bất thường, dị biệt không thể suy luận từ quy tắc chung.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. **Chép một record:**
   - `instance_id` = `kitchen-001`
   - `class_name` = `person`
   - `score` = `0.899318`
   - Số điểm polygon = `polygon_point_count` = `348`
   - 3–5 điểm đầu của `polygon_xy` = `[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]`

2. **So với box, polygon cho thêm thông tin gì về hình dạng vật thể?**
   - Bounding box chỉ cung cấp một khung chữ nhật bao ngoài, chứa lẫn cả không gian nền và các vật thể phụ nằm xen vào các góc trống.
   - **Polygon (Đa giác phân đoạn)** cung cấp thông tin hình học chi tiết chính xác đến từng pixel về: **đường bao thực tế (contour/boundary), diện tích thực của vật thể (true pixel area), độ cong, hình dáng thực tế** (như chi tiết đầu, cánh tay, dáng đứng của người) và tách biệt hoàn toàn vật thể khỏi phông nền.

3. **`instance_id` dùng để làm gì? Nó không phải class ID và không phải tracking ID — bạn hiểu câu này thế nào?**
   - `instance_id` (ví dụ `kitchen-001`, `kitchen-002`) là định danh duy nhất dùng để **phân biệt từng cá thể đối tượng độc lập trong cùng một bức ảnh**, ngay cả khi chúng có cùng lớp ngữ nghĩa (ví dụ trong ảnh có nhiều người hoặc nhiều cái bát thì mỗi cái có một `instance_id` riêng).
   - **Không phải class ID:** Vì class ID chỉ định loại đối tượng chung trong taxonomy (ví dụ tất cả các người đều chung `class_id = 0`); class ID không phân biệt người A với người B.
   - **Không phải tracking ID:** Vì tracking ID là mã duy trì danh tính một đối tượng xuyên suốt một chuỗi video nhiều khung hình liên tiếp theo thời gian. `instance_id` ở đây chỉ có ý nghĩa phân định cục bộ trong một bức ảnh tĩnh duy nhất của bài lab.

4. **Viết một quy tắc vẽ biên mask để người gán nhãn làm theo:**
   - **Quy tắc đường viền mask chính xác (Precise Polygon Boundary Rule):**  
     > *"Đa giác mask phải đi sát theo đường mép tiếp giáp thực tế của vật thể theo chiều kim đồng hồ, độ sai lệch biên tối đa không quá 1-2 pixel; tại các góc bo tròn hoặc chi tiết phức tạp cần phân bố dày điểm nút để đường biên liền mạch, tuyệt đối không khoét lẹm vào trong vật thể và không gộp phần nền vào trong mặt nạ."*

5. **Trên ảnh `kitchen`, chỉ ra một chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?**
   - **Chỗ quan sát:** Khu vực các tô bát đặt sát nhau trên mặt bàn hoặc vị trí tiếp xúc giữa đáy tô bát (`bowl`) với mặt bàn đá bếp nơi có bóng đổ và ánh sáng phản xạ làm ranh giới pixel bị mờ nhạt.
   - **Lựa chọn:** Dùng **Guideline** cho các trường hợp chạm nhau thông thường (quy định: *"Khi hai vật chạm nhau, vẽ ranh giới tách rời theo đường phân định màu sắc/vật liệu rõ nhất; vật ở trước được vẽ đầy đủ, vật ở sau chỉ vẽ phần hiển thị"*).
   - **Khi nào dùng Escalation:** Nếu gặp vùng ranh giới bị bóng đen che lấp hoàn toàn đến mức mắt thường không thể xác định đâu là đáy bát và đâu là mặt bàn, người gán nhãn cần gắn cờ **Escalation** gửi Lab Coach để thống nhất tiêu chuẩn gán nhãn đồng nhất cho toàn đội ngũ, tránh mỗi người đoán một kiểu.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | 1 nhãn lớp (class label) duy nhất (hoặc tập hợp nhãn multi-label) đại diện cho toàn bộ bức ảnh | Ảnh `traffic` có nhiều đối tượng cùng xuất hiện (xe taxi, minibus, người, phố sá). Model xếp hạng 1 là `cab` (score 0.51), hạng 2 là `minibus` (score 0.16) gây khó khăn khi chọn một lớp duy nhất cho toàn ảnh | Đọc kỹ guideline, xác định đối tượng chiếm ưu thế/tiền cảnh để chọn chính xác 1 nhãn lớp phù hợp nhất trong taxonomy ImageNet-1K | Kiểm tra sự tuân thủ quy tắc chọn chủ thể của annotator; đối chiếu độ đồng thuận nhãn giữa các annotator (inter-annotator agreement) |
| Phát hiện vật thể | Tập hợp các bounding box hình chữ nhật `[x_min, y_min, x_max, y_max]` kèm `class_id` cho từng vật thể | Trên ảnh `kitchen`, các vật nhỏ hoặc bị che khuất như cái thìa (`spoon`) ở ngưỡng 0.35 bị bỏ sót; hạ ngưỡng xuống 0.20 thì xuất hiện nhiều box thừa | Vẽ bounding box chuẩn xác ôm khít từng đối tượng có trong taxonomy COCO-80, gán đúng nhãn lớp, không bỏ sót vật thể | Soi độ ôm sát (IoU) của box, kiểm tra không bị cắt lẹm vào vật thể, phát hiện các vật bị bỏ sót (False Negative) hoặc gán nhầm lớp (False Positive) |
| Instance segmentation | Tập hợp các đa giác kín `polygon_xy` bám sát đường biên pixel của từng cá thể đối tượng kèm `class_id` và `instance_id` | Ranh giới đáy bát chạm mặt bàn bị bóng đổ mờ; đường viền cơ thể người đầu bếp chạm vào bàn bếp gây khó khăn khi xác định tọa độ biên chính xác | Sử dụng công cụ vẽ đa giác tỉ mỉ bám sát viền ngoài của từng instance riêng biệt; tách rời các vật cùng lớp thành các cá thể độc lập | Phóng to kiểm tra độ chính xác của đường viền (boundary accuracy), kiểm tra các vùng rỗng (holes), đảm bảo không vẽ lấn sang đối tượng khác |

## 5. An toàn dữ liệu

1. **Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu):**
   - *"Chỉ sử dụng hình ảnh công khai từ tập dữ liệu COCO và checkpoint chuẩn đã được xác thực mã băm SHA-256; tuyệt đối không đưa hình ảnh cá nhân, thông tin định danh (họ tên, MSSV, SĐT, email) hay dữ liệu nhạy cảm vào mã nguồn, artifact và báo cáo."*

2. **Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?:**
   - Nếu phát hiện dữ liệu nằm ngoài phạm vi cho phép (chứa thông tin cá nhân, hình ảnh riêng tư hoặc dữ liệu bảo mật), tôi sẽ **lập tức dừng xử lý và báo cáo ngay cho Lab Coach / Giảng viên phụ trách lớp** để được hướng dẫn xử lý và cô lập dữ liệu an toàn.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS

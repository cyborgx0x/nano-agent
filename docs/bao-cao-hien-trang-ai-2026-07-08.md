# Báo cáo hiện trạng triển khai AI của nano-agent

**Ngày lập:** 2026-07-08
**Phạm vi:** Toàn bộ mã nguồn và tài liệu trong kho mã nguồn nano-agent tại thời điểm lập báo cáo.
**Cách thực hiện:** Đọc mã nguồn và tài liệu trực tiếp. Chưa chạy được mã do môi trường thiếu thư viện phụ thuộc.

## 1. Mục tiêu dùng để đối chiếu

Dự án đặt ra 3 tầng mục tiêu cho phần trí tuệ nhân tạo.

Tầm nhìn tổng thể ghi trong `README.md` là một chương trình tự động chơi Albion Online, và chỉ được quan sát trò chơi qua ảnh chụp màn hình.

Mục tiêu của bản chạy tối thiểu khả dụng ghi trong `docs/problem hypothesis.md` là một Vòng lặp Sinh tồn và Kinh tế. Trong vòng lặp này, tác tử farm tài nguyên ở vùng an toàn, né sinh vật để giảm rủi ro tử vong, farm sinh vật mức nhẹ để lấy điểm kinh nghiệm, rồi quay về thành cất tài nguyên. Tài liệu chủ trương huấn luyện trong trình mô phỏng trạng thái trước, theo lộ trình từ giai đoạn 0 đến giai đoạn 7. Điều kiện để tác tử lên giai đoạn kế tiếp là tỉ lệ thành công đạt từ 0,85 trở lên và tỉ lệ tử vong không quá 0,05.

Bộ tiêu chí chất lượng ghi trong `docs/rl_world_class_checklist.md` định nghĩa thế nào là một hệ thống học tăng cường tiệm cận tầm thế giới, kèm một thang tự chấm từ 0 đến 14 điểm.

## 2. Hiện trạng tổng quan

Mã nguồn hiện chứa 4 hướng tiếp cận trí tuệ nhân tạo cùng tồn tại song song, và không hướng nào đã đạt tới trạng thái hoàn chỉnh. Bảng dưới đây tóm tắt 4 hướng đó.

| Hướng tiếp cận | Vị trí trong mã nguồn | Quy mô | Trạng thái |
| --- | --- | --- | --- |
| Chương trình phản xạ nhận dạng đối tượng | `agent/`, `components/`, `main.py` | Khoảng 1.000 dòng cộng máy chủ suy luận Docker | Có tương tác trò chơi thật nhưng thuần theo luật cứng, không có học |
| Học tăng cường trong trình mô phỏng | `state_sim/` | Khoảng 4.200 dòng | Được đầu tư nhiều nhất, nhưng chưa hội tụ và chưa đạt tiêu chí lên giai đoạn |
| Mô hình thế giới V-JEPA | `world_model/` | Khoảng 1.400 dòng | Mới dừng ở dựng khung, chưa thu thập dữ liệu, chưa huấn luyện |
| Các thử nghiệm thăm dò khác | `slam/`, `prototype/`, `spatial_world_model.py` | Rải rác | Thử nghiệm rời rạc, chưa gắn vào một trục chính |

## 3. Hiện trạng chi tiết theo từng hướng

### Chương trình phản xạ nhận dạng đối tượng

Đây là hướng duy nhất thực sự tương tác với trò chơi thật. Chương trình chụp màn hình, chạy mô hình nhận dạng đối tượng để tìm tài nguyên, rồi bấm chuột vào tài nguyên. Bản chất của hướng này là phản xạ theo luật cứng, không có thành phần học tự cải thiện. Tệp `agent/agent.py`, nơi lẽ ra chứa tác tử có trạng thái theo hướng học tăng cường, tự khai báo ở phần đầu rằng đây mới là khung thiết kế chưa hoàn chỉnh và hiện chưa được dùng.

### Học tăng cường trong trình mô phỏng

Đây là hướng đúng với chủ trương huấn luyện trong trình mô phỏng trước, và cũng là hướng có mã trưởng thành nhất, với quy mô khoảng 4.200 dòng. Hướng này có môi trường mô phỏng theo vùng và quần xã sinh vật, có lộ trình huấn luyện, có thuật toán tối ưu chính sách gần đúng với kiến trúc pha trộn chuyên gia, có cơ chế giáo viên dẫn dắt, và có bước đánh giá trên tập vùng giữ lại để đo khả năng tổng quát hóa.

Tuy nhiên, tài liệu `docs/component_analysis.md` lập ngày 2026-03-01 tự ghi nhận hệ thống này chưa hội tụ. Tỉ lệ thành công khi huấn luyện dao động trong khoảng từ 33% đến 68%. Tỉ lệ thành công trên tập vùng giữ lại bằng 0, tức tác tử thất bại hoàn toàn khi gặp vùng chưa từng huấn luyện. Độ dài mỗi lượt chơi tăng từ 286 bước lên 1.169 bước, cho thấy tác tử chậm dần thay vì hiệu quả hơn. So với điều kiện lên giai đoạn là tỉ lệ thành công từ 0,85 trở lên, hệ thống này chưa vượt qua được các giai đoạn đầu của lộ trình một cách ổn định.

Khi đối chiếu tài liệu phân tích với mã nguồn hiện tại, tôi nhận thấy các khuyến nghị sửa lỗi mới được áp dụng một phần. Khuyến nghị ưu tiên số 1 là loại bỏ mã hóa định danh vùng đã được áp dụng, vì tệp `state_sim/ppo/encoder.py` hiện dùng thuộc tính loại vùng và quần xã sinh vật thay cho định danh vùng. Khuyến nghị ưu tiên số 2 là giảm ngưỡng giáo viên tối thiểu về gần 0 thì chưa được áp dụng, vì tệp `state_sim/ppo/config.py` vẫn để ngưỡng này ở mức 0,35. Khuyến nghị ưu tiên số 3 là đánh giá tập vùng giữ lại bằng cách lấy mẫu thay cho lấy giá trị lớn nhất thì cũng chưa được áp dụng, vì tệp `state_sim/ppo/trainer.py` vẫn dùng phép lấy giá trị lớn nhất.

Vì mới sửa 1 trong 3 nguyên nhân gốc rễ, và vì trong kho mã nguồn không có thư mục lưu kết quả huấn luyện, không có tệp checkpoint là tệp lưu trạng thái mô hình tại một thời điểm, không có nhật ký thực nghiệm nào được lưu lại, hiện chưa có bằng chứng nào xác nhận rằng tỉ lệ thành công trên tập vùng giữ lại đã thoát khỏi mức 0 sau khi sửa phần mã hóa.

### Mô hình thế giới V-JEPA

Hướng này có quy mô khoảng 1.400 dòng mã. Tài liệu `docs/WORLD_MODEL_SUMMARY.md` tự đánh dấu giai đoạn 1 đã hoàn thành. Cần hiểu chính xác rằng giai đoạn 1 ở đây chỉ có nghĩa là đã sao chép kiến trúc từ mã nguồn gốc của Meta và viết khung tích hợp cho việc điều khiển chuột và bàn phím. Bản thân tài liệu liệt kê các giai đoạn tiếp theo, gồm thu thập từ 10 đến 20 giờ dữ liệu chơi, tải trọng số đã huấn luyện sẵn, huấn luyện tinh chỉnh, và kiểm thử vòng kín, và cả 4 giai đoạn này đều chưa bắt đầu. Nói cách khác, đây là một bộ khung chưa từng chạy một vòng học nào, chưa có dữ liệu và chưa có mô hình.

### Các thử nghiệm thăm dò khác

Nhóm này gồm điều hướng theo phương pháp định vị và dựng bản đồ đồng thời trong `slam/`, bản mẫu trong `prototype/`, và mô hình thế giới không gian trong `state_sim/spatial_world_model.py`. Có 2 thư mục là `research/` và `secbox/`, cả hai đều là mô đun con chưa được khởi tạo nên hiện đang rỗng.

## 4. Đối chiếu với mục tiêu

Xét theo lộ trình của bản chạy tối thiểu khả dụng, dự án mới chạm tới các giai đoạn đầu là điều hướng trong vùng và gom tài nguyên, nhưng chưa đạt điều kiện lên giai đoạn vì tỉ lệ thành công còn thấp và khả năng tổng quát hóa còn bằng 0. Các giai đoạn sau, gồm chuyển vùng, vòng kinh tế đầy đủ, và tích hợp điểm kinh nghiệm, đều chưa được chạm tới. Cầu nối từ chính sách trong mô phỏng sang điều khiển trò chơi thật, gồm bộ lọc an toàn và cơ chế dự phòng, cũng chưa được khởi động.

Xét theo thang tự chấm mức độ sẵn sàng trong `docs/rl_world_class_checklist.md`, phần lớn tiêu chí đều chưa đạt. Khả năng tái lập chưa có vì không có checkpoint hay lệnh chạy lại được lưu. Độ ổn định qua nhiều hạt giống ngẫu nhiên chưa có báo cáo. Khả năng tổng quát hóa đang ở mức bằng 0 theo tài liệu phân tích. Với hiện trạng này, tổng điểm tự chấm ước tính nằm trong khoảng từ 0 đến 3 trên 14, tức thuộc mức nguyên mẫu theo chính thang đo của dự án.

## 5. Các vấn đề chính

Vấn đề thứ nhất là nỗ lực bị phân tán. 4 hướng tiếp cận đang chạy song song và làm nguồn lực bị dàn trải, khiến không hướng nào được đẩy tới trạng thái chạy được và đo được.

Vấn đề thứ hai là vòng chẩn đoán chưa được khép kín. Đội ngũ đã chẩn đoán đúng nguyên nhân khiến khả năng tổng quát hóa bằng 0, nhưng mới sửa 1 trong 3 nguyên nhân và chưa chạy lại để kiểm chứng, trong khi 2 sửa đổi được đánh giá là bắt buộc vẫn còn nguyên trong mã.

Vấn đề thứ ba là chưa có tài sản thực nghiệm nào được lưu lại. Kho mã nguồn không có checkpoint, không có thư mục kết quả huấn luyện, và không có báo cáo số liệu. Đây chính là ranh giới phân biệt giữa một nguyên mẫu và một hệ thống bán chuyên.

## 6. Việc nên làm tiếp

Việc nên làm trước tiên là chọn một hướng làm trục chính. Theo đúng tài liệu mục tiêu, hướng đó phải là học tăng cường trong trình mô phỏng.

Việc có giá trị cao ngay sau đó là áp nốt 2 sửa đổi còn thiếu trong `state_sim/ppo/config.py` và `state_sim/ppo/trainer.py`, rồi chạy một đợt huấn luyện qua nhiều hạt giống ngẫu nhiên, lưu lại checkpoint và số liệu trên tập vùng giữ lại, để xác nhận vấn đề gốc rễ đã được giải quyết.

Việc cần làm để vượt khỏi mức nguyên mẫu là thiết lập nơi lưu kết quả thực nghiệm và một lệnh chạy lại được, đúng như phần sản phẩm bàn giao mà bộ tiêu chí chất lượng đã yêu cầu.

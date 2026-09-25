# RL Training Playbook (General)

Mục tiêu: một quy trình chung để **vừa học RL bài bản**, vừa **triển khai thực tế** cho bất kỳ use case nào (game, robot, trading, control).

## 1) Tư duy nền tảng trước khi code
Trước khi train, phải chốt rõ 5 thành phần:
- **Task**: bài toán gì, tiêu chí thành công là gì.
- **State/Observation**: agent được nhìn thấy gì.
- **Action Space**: agent có thể làm gì.
- **Reward**: khuyến khích hành vi nào, phạt hành vi nào.
- **Episode/Termination**: khi nào kết thúc 1 lượt chạy.

Nếu 1 trong 5 phần mơ hồ, training sẽ khó ổn định.

## 2) Quy trình học RL theo pha

### Pha A - Foundation (hiểu và test nhanh)
1. Học MDP cơ bản, exploration/exploitation, value vs policy.
2. Chạy vài môi trường chuẩn (CartPole, LunarLander, Pendulum).
3. Nắm 3 họ thuật toán:
   - Value-based: DQN (discrete).
   - Policy gradient: REINFORCE/PPO.
   - Actor-Critic: A2C/SAC/TD3.
4. Học cách đọc learning curve và debug reward.

Kết quả pha A: bạn biết chọn thuật toán đúng loại action space và nhận ra dấu hiệu train hỏng.

### Pha B - Environment-first (cho use case thực tế)
1. Xây môi trường chuẩn `reset/step`.
2. Viết unit test cho env (state shape, done condition, reward range).
3. Tạo baseline không học (rule-based hoặc heuristic).
4. Log đầy đủ metrics ngay từ đầu.

Kết quả pha B: môi trường đúng và có baseline để so sánh.

### Pha C - Training loop chuẩn
1. Chạy thử với cấu hình nhỏ để debug.
2. Tune tối thiểu: learning rate, gamma, batch size, entropy.
3. Train nhiều seed khác nhau để kiểm tra độ ổn định.
4. Lưu checkpoint theo best metric trên validation.

Kết quả pha C: có policy vượt baseline trong điều kiện kiểm soát.

### Pha D - Generalization & Productionization
1. Domain randomization (nhiễu observation, random spawn, random map).
2. Curriculum từ dễ -> khó.
3. Tách tập validation/test không trùng train.
4. Thêm guardrail khi deploy (fallback rule-based).

Kết quả pha D: policy bền vững khi chạy thực tế.

## 3) Dataset có cần không?
Ngắn gọn:
- **Không bắt buộc** nếu train RL online thuần.
- **Rất nên có** nếu môi trường tốn chi phí, reward thưa, hoặc hành vi khó học.

### Khi nên thu thập dataset trước
- Tương tác thật tốn thời gian/chi phí.
- Có thể lấy dữ liệu từ chuyên gia hoặc bot heuristic.
- Muốn warm-start để giảm thời gian khám phá ngẫu nhiên.

### 3 cách dùng dataset trong RL
1. **Behavior Cloning (BC)**: học bắt chước ban đầu.
2. **Offline RL**: học policy từ batch dữ liệu tĩnh.
3. **Hybrid**: pretrain từ dataset rồi fine-tune online.

Trong đa số dự án thực tế, hybrid là lựa chọn cân bằng tốt nhất.

## 4) Framework ra quyết định thuật toán
- Action rời rạc, state vừa phải: bắt đầu với `PPO` hoặc `DQN`.
- Action liên tục: ưu tiên `SAC` hoặc `TD3`.
- Observation ảnh: dùng CNN encoder + PPO/SAC.
- Reward thưa: thêm shaping, HER, hoặc curriculum.

Quy tắc thực dụng: chọn thuật toán ổn định và dễ debug trước, chưa cần tối ưu SOTA ngay.

## 5) Bộ metrics chung bắt buộc
- `Success Rate`.
- `Average Return`.
- `Sample Efficiency` (mất bao nhiêu bước để đạt mốc).
- `Stability` (độ lệch giữa nhiều seed).
- `Generalization Score` trên test ngoài phân phối train.

## 6) Chu trình làm việc hằng tuần
1. Chốt 1 thay đổi nhỏ (reward, state, hyperparam, architecture).
2. Chạy thí nghiệm có kiểm soát (giữ các biến còn lại cố định).
3. So sánh với baseline bằng cùng metric.
4. Ghi log + kết luận rõ: giữ, rollback, hay thử tiếp.

Không đổi nhiều thứ cùng lúc, nếu không sẽ không biết yếu tố nào tạo cải thiện.

## 7) Các lỗi phổ biến khi học RL
- Reward hack: agent tối ưu reward sai mục tiêu thật.
- Overfit môi trường train: qua map mới thì thất bại.
- Không tách validation/test: tưởng tốt nhưng không tổng quát.
- Tune thiếu kỷ luật: thay quá nhiều tham số một lúc.
- Không có baseline: không biết policy học có thực sự tốt hơn chưa.

## 8) Mốc năng lực (learning milestones)
- **Mốc 1**: tự build env + train PPO/DQN chạy được.
- **Mốc 2**: debug được khi curve không tăng.
- **Mốc 3**: policy vượt baseline ổn định qua nhiều seed.
- **Mốc 4**: policy chạy được ở điều kiện thực tế có nhiễu.
- **Mốc 5**: có pipeline tái lập (reproducible) từ data đến deploy.

## 9) Checklist trước khi deploy thật
- [ ] Có baseline và policy đã vượt baseline rõ ràng.
- [ ] Kết quả ổn định qua ít nhất 3 random seeds.
- [ ] Có test ngoài phân phối train.
- [ ] Có fallback khi policy hành vi bất thường.
- [ ] Có logging đủ để điều tra lỗi sau deploy.

---

## Kết luận ngắn
- RL **không bắt buộc dataset**, nhưng dataset giúp học nhanh và an toàn hơn trong bài toán thực tế.
- Lộ trình tốt nhất để học và làm thật: `Foundation -> Environment -> Baseline -> (Optional Dataset/Warm Start) -> Online RL -> Generalization -> Deployment`.

# RL Agent Albion Online — Checklist từ Prototype tới "World-Class"

Mục tiêu của checklist này: biến một bản train chạy được thành một hệ thống RL có chất lượng nghiên cứu/kỹ thuật cao, có thể tái lập và mở rộng.

## 0) Definition of Done (định nghĩa “đã tiệm cận tầm thế giới”)

- [ ] Có **benchmark rõ ràng** (task, map, seed, thời lượng, metric).
- [ ] Kết quả **ổn định qua nhiều seed** (không chỉ một run đẹp).
- [ ] Agent **generalize** sang map/tình huống mới với suy giảm chấp nhận được.
- [ ] Có **ablation study** chứng minh thành phần nào thực sự có ích.
- [ ] Toàn bộ pipeline **reproducible** trên máy khác.
- [ ] Có báo cáo tổng hợp (W&B + markdown) đủ để người khác review.

---

## 1) Data & Environment Hygiene

- [ ] Chốt phiên bản env/simulator, map set, action/state spec.
- [ ] Tách rõ `train/val/test maps` (không leakage).
- [ ] Chuẩn hóa reward scale (tránh reward quá lớn/nhỏ gây bất ổn PPO).
- [ ] Có kiểm tra deterministic tối đa có thể (seed env, numpy, torch).
- [ ] Log đầy đủ episode trajectory cơ bản (return, length, success/fail).
- [ ] Kiểm tra distribution shift giữa map train và map test.

**Deliverable:** tài liệu mô tả environment + file config cố định cho thực nghiệm chính.

---

## 2) Baseline mạnh trước khi tối ưu

- [ ] Thiết lập baseline heuristic/rule-based tối thiểu (để biết RL có thực sự hơn không).
- [ ] Thiết lập baseline RL đơn giản (PPO default) làm mốc.
- [ ] Chốt metric chính:
  - [ ] Success rate
  - [ ] Mean return
  - [ ] Steps-to-goal / efficiency
  - [ ] Safety metric (kẹt, va chạm, timeout)
- [ ] Vẽ learning curve có smoothing + raw để tránh ngộ nhận.

**Deliverable:** bảng so sánh baseline trong cùng ngân sách compute.

---

## 3) Training Stability (quan trọng nhất)

- [ ] Chạy tối thiểu 3–5 seed cho mỗi cấu hình chính.
- [ ] Theo dõi chỉ số ổn định PPO:
  - [ ] Policy loss
  - [ ] Value loss
  - [ ] Entropy
  - [ ] KL divergence / clip fraction
- [ ] Bật gradient clipping, kiểm soát learning rate schedule.
- [ ] Kiểm tra reward clipping/normalization nếu cần.
- [ ] Early-stop khi diverge rõ ràng để tiết kiệm compute.

**Deliverable:** biểu đồ mean ± std theo seed.

---

## 4) Generalization & Robustness

- [ ] Hold-out map test: không dùng trong train.
- [ ] Domain randomization (spawn, nhiễu quan sát, timing variance).
- [ ] Stress test với tình huống hiếm (bị kẹt, path blocked, object missing).
- [ ] Đánh giá OOD score: hiệu năng trên map mới so với map train.
- [ ] Kiểm tra khả năng recovery khi state bị lệch.

**Deliverable:** báo cáo train-vs-test gap + các failure mode chính.

---

## 5) Reward & Credit Assignment

- [ ] Tách reward thành các thành phần và log riêng từng thành phần.
- [ ] Kiểm tra agent có exploit reward không (reward hacking).
- [ ] So sánh sparse vs shaped reward (ít nhất 1 thử nghiệm).
- [ ] Nếu dùng curriculum: định nghĩa điều kiện lên level rõ ràng.
- [ ] Kiểm chứng reward shaping không phá objective cuối.

**Deliverable:** reward card (mô tả + lý do + ảnh hưởng quan sát được).

---

## 6) Ablation Study (điểm phân biệt “pro”)

- [ ] Ablation action masking (on/off).
- [ ] Ablation state feature groups (vision/path/state history).
- [ ] Ablation curriculum (with/without).
- [ ] Ablation architecture (MLP vs LSTM/Transformer nếu có).
- [ ] Ablation augmentation/domain randomization.

**Deliverable:** bảng ablation có delta metric + chi phí train.

---

## 7) System & MLOps

- [ ] Cấu trúc experiment config thống nhất (yaml/toml + run id).
- [ ] Lưu checkpoint định kỳ + best checkpoint theo metric.
- [ ] Auto resume run khi interrupted.
- [ ] Version hóa dataset/map/config/model.
- [ ] W&B dashboard chuẩn hóa (naming, tags, groups, notes).
- [ ] Tạo script một lệnh để reproduce run chính.

**Deliverable:** “Reproduce in 1 command” trong README.

---

## 8) Offline → Online bridge (Albion thực chiến)

- [ ] Mapping từ action policy sang input thực tế (an toàn, rate-limited).
- [ ] Bộ lọc safety trước khi gửi action (hard constraints).
- [ ] Cơ chế fallback heuristic khi model uncertain.
- [ ] Human-in-the-loop mode để can thiệp khi fail.
- [ ] Logging online tách biệt để phân tích lỗi deployment.

**Deliverable:** checklist triển khai an toàn + rollback plan.

---

## 9) Đánh giá “World-Class Readiness” (score nhanh)

Chấm mỗi mục 0–2:
- 0 = chưa làm
- 1 = có làm nhưng chưa ổn định/thiếu bằng chứng
- 2 = hoàn chỉnh, có log + báo cáo

- [ ] Reproducibility
- [ ] Stability across seeds
- [ ] Generalization
- [ ] Ablation quality
- [ ] Baseline fairness
- [ ] MLOps maturity
- [ ] Deployment safety

**Tổng điểm (0–14):**
- 0–5: Prototype / local strong
- 6–10: Semi-pro / rất tốt trong cộng đồng
- 11–14: Tiệm cận world-class engineering-research

---

## 10) Kế hoạch 30 ngày gợi ý (thực dụng)

### Tuần 1 — Ổn định pipeline
- [ ] Chốt metric + dashboard + seed policy.
- [ ] Thiết lập baseline heuristic + PPO default.
- [ ] Chạy 3 seed đầu tiên và sửa lỗi stability.

### Tuần 2 — Generalization
- [ ] Tạo split map train/val/test sạch.
- [ ] Bổ sung domain randomization tối thiểu.
- [ ] Báo cáo gap train-test đầu tiên.

### Tuần 3 — Ablation
- [ ] Chạy 3–4 ablation quan trọng nhất.
- [ ] Tổng hợp bảng kết quả + chi phí compute.

### Tuần 4 — Đóng gói & công bố nội bộ
- [ ] 1-command reproduce.
- [ ] Viết report ngắn: phương pháp, kết quả, giới hạn, hướng tiếp.
- [ ] Chốt model candidate cho online thử nghiệm có safety guard.

---

## Notes cho repo hiện tại

- Bạn đã có nền tảng tốt: `state_sim/`, script train, và W&B offline sync.
- Bước tiếp theo đáng làm ngay: chuẩn hóa **multi-seed runs + hold-out map test + ablation tối thiểu**.
- Khi 3 mảng đó tốt, bạn đã vượt xa “ao làng” và tiến gần chuẩn quốc tế.

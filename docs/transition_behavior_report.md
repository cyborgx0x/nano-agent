# Báo cáo: Hành vi agent khi di chuyển từ bản đồ này sang bản đồ khác

**Date:** 2026-05-22
**Status:** ✅ Hoàn thành
**Epic:** [#1](https://github.com/cyborgx0x/nano-agent/issues/1) — sub-issues #2, #3, #4, #5

---

## 1. Mục tiêu

Đo và so sánh **hành vi của các kiến trúc agent khác nhau tại thời điểm chuyển bản đồ**
(gate transition giữa các zone). Câu hỏi nghiên cứu:

1. Kiến trúc nào phục hồi định hướng nhanh nhất sau khi vào zone mới?
2. Full view có cải thiện quyết định ngay tại gate không?
3. Memory (zone graph / NN weights) giúp ích như thế nào trong transition?
4. Stagnation sau transition — kiến trúc nào bị ảnh hưởng nhiều nhất?

---

## 2. Thiết lập thực nghiệm

### Môi trường

`AlbionStateSim` — zone graph 36 zone, agent di chuyển qua gate. Hai bổ sung
phục vụ nghiên cứu này (issue #2):

- `set_observation_mode("partial" | "full")` — chuyển giữa observed view
  (vision radius 0.30) và full view (toàn bộ zone).
- `info["transition_event"]` — phát sự kiện `{from_zone, to_zone, step}` mỗi
  khi agent crossing gate.

**Phát hiện về cấu trúc map:** zone graph gồm **5 thành phần liên thông tách rời**
(forest / steppe / mountain / swamp / snow, mỗi cluster gắn một city). Task
navigation chỉ giải được khi start và goal cùng component — benchmark vì vậy
chỉ sample task cùng component (`solvable_tasks()`).

### Kiến trúc agent (issue #3 — `state_sim/agent_interface.py`)

Tất cả tuân theo `AgentProtocol` (`reset` / `act` / `observe` / `name`):

| Agent | Kiểu | Đặc điểm liên quan đến transition |
|-------|------|-----------------------------------|
| `random` | Baseline | Action ngẫu nhiên — sàn so sánh |
| `greedy` | Heuristic | BFS trên zone graph → đi tới gate đúng. Có "memory hoàn hảo" (zone graph tĩnh) |
| `tabular_q` | Tabular RL | Q-table trên feature egocentric rời rạc hoá. **Không** có function approximation, **không** biết zone graph |
| `ppo` | Deep RL | Actor-critic feedforward (~0.2M params), checkpoint train ngắn (1.3M steps) |

### Benchmark (issue #4 — `state_sim/transition_benchmark.py`)

- 200 task navigation cùng component, mỗi kiến trúc × 2 observation mode.
- Window đo = 10 bước sau mỗi transition event.
- Lệnh tái lập:

```bash
python -m state_sim.transition_benchmark \
    --agents random greedy tabular_q ppo \
    --episodes 200 --observation-mode both \
    --tabular-train-episodes 3000 --window 10 --seed 123 \
    --output-csv results/transition_metrics.csv
```

---

## 3. Kết quả

| agent | obs_mode | nav_success | transitions/ep | dwell_steps | displacement | stagnation | goal_progress | backtrack_rate | first_entropy | n_transitions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random | partial | 0.03 | 0.25 | 100.3 | 0.914 | 0.31 | 0.000 | 0.33 | — | 50 |
| greedy | partial | **0.98** | 2.06 | 8.6 | 0.684 | 0.55 | **0.364** | **0.00** | — | 412 |
| tabular_q | partial | 0.01 | 0.02 | 92.8 | 0.734 | 0.93 | 0.000 | — | — | 4 |
| ppo | partial | 0.24 | 5.39 | 38.4 | 1.174 | 21.99 | 0.005 | 0.85 | 2.287 | 1079 |
| random | full | 0.03 | 0.23 | 81.9 | 0.996 | 0.18 | 0.000 | 0.50 | — | 47 |
| greedy | full | **0.99** | 2.08 | 8.7 | 0.678 | 0.73 | **0.351** | **0.00** | — | 416 |
| tabular_q | full | 0.15 | 15.80 | 9.9 | 0.168 | 102.61 | -0.010 | 1.00 | — | 3160 |
| ppo | full | 0.26 | 4.22 | 46.9 | 1.051 | 18.56 | 0.001 | 0.77 | 2.363 | 845 |

Ý nghĩa metric: `dwell_steps` = số bước ở lại zone trước khi chuyển tiếp;
`displacement` = quãng đường đi trong window 10 bước; `stagnation` = bộ đếm
đứng yên trung bình; `goal_progress` = mức giảm khoảng cách BFS tới goal sau
window (dương = tiến bộ); `backtrack_rate` = tỉ lệ transition kế tiếp quay lại
đúng zone vừa rời (đo mất định hướng / dao động).

---

## 4. Trả lời câu hỏi nghiên cứu

### Q1 — Kiến trúc nào phục hồi định hướng nhanh nhất sau khi vào zone mới?

**`greedy`, áp đảo.** `dwell_steps ≈ 8.6` và `goal_progress ≈ +0.36`: ngay khi
vào zone mới, agent chạy lại BFS, xác định gate đúng và tiến tới goal trong mỗi
transition. `backtrack_rate = 0.00` — không bao giờ quay lại nhầm.

- `ppo`: `dwell ≈ 38–47`, `goal_progress ≈ 0.00` — chậm và **không** có tiến
  bộ định hướng; mỗi transition gần như không rút ngắn đường tới goal.
- `tabular_q`: không bao giờ phục hồi (xem Q4).

### Q2 — Full view có cải thiện quyết định tại gate không?

**Phần lớn là KHÔNG.** Nút thắt nằm ở kiến trúc/chính sách, không phải ở mức
độ quan sát:

- `greedy`: kết quả gần như giống hệt partial vs full (0.98 vs 0.99) — vì
  greedy dùng zone graph, không phụ thuộc vision.
- `ppo`: 0.24 → 0.26, cải thiện không đáng kể.
- `tabular_q`: full view **không** tạo ra navigation có năng lực — nó chỉ
  *đổi kiểu thất bại* (xem Q4). Success tăng 0.01 → 0.15 thuần tuý do dao động
  ngẫu nhiên thỉnh thoảng trúng goal.

→ Cung cấp thêm thông tin quan sát không tự động sửa được hành vi transition.

### Q3 — Memory giúp ích như thế nào?

So sánh trực tiếp ba mức "memory":

- `greedy` — **memory tĩnh hoàn hảo** (zone graph): `backtrack = 0.00`, luôn
  biết gate nào dẫn về đâu.
- `ppo` — memory trong **NN weights** (kỹ năng tổng quát hoá): tốt hơn tabular
  (24% vs 1–15%) nhờ chính sách trơn trên feature liên tục, nhưng **không** có
  episodic memory về gate vừa đi qua → `backtrack 0.77–0.85`.
- `tabular_q` — **không** memory, state rời rạc không mang hướng-tới-goal →
  `backtrack 1.00` ở full mode.

**Phát hiện cốt lõi:** thiếu memory về *gate vừa dùng để vào zone* là nguyên
nhân trực tiếp của dao động sau transition. Agent "quên" mình vừa từ đâu tới và
lập tức đi lại vào gate đó.

### Q4 — Stagnation sau transition — kiến trúc nào bị ảnh hưởng nhiều nhất?

**`tabular_q`, nghiêm trọng nhất**, với hai kiểu bệnh lý khác nhau tuỳ
observation mode:

- **Partial:** `stagnation 0.93`, `transitions/ep 0.02` (chỉ 4 transition trên
  toàn bộ 200 episode!). Agent **không thấy gate** (vision 0.30) → lang thang
  và gần như không bao giờ qua được gate. Kiểu thất bại = "explorer đóng băng".
- **Full:** `stagnation 102.6`, `displacement 0.168` (gần như bất động),
  `backtrack 1.00`, `transitions/ep 15.8`. Agent **luôn thấy gate** → bám lấy
  một gate và bật qua bật lại. Kiểu thất bại = "dao động bám gate".

`ppo`: stagnation trung bình (~19–22) — có dao động nhưng vẫn di chuyển nhiều
(`displacement ≈ 1.0–1.2`). `greedy`: stagnation không đáng kể (0.55–0.73).

---

## 5. Phân tích hành vi từng kiến trúc

**greedy** — Chuẩn tham chiếu. 2.06 transition/episode = đi đúng đường ngắn
nhất. Tái định hướng tức thì, tiến bộ đều, không dao động. Bất biến theo
observation mode. Hạn chế: là heuristic, cần zone graph cho trước, không học.

**tabular_q** — Minh hoạ rõ giới hạn của tabular RL không function
approximation: state rời rạc egocentric không thể mã hoá "gate nào dẫn tới
goal". Đáng chú ý — **thêm thông tin (full view) làm hành vi tệ đi**: chuyển
từ đóng băng sang dao động bám gate (`backtrack 1.00`). Đây là cảnh báo: với
agent memoryless, quan sát đầy đủ hơn có thể gây bệnh lý mới.

**ppo** — Checkpoint train ngắn (~1.3M steps, `nav_success` lúc train ≈ 1%; xem
ghi chú bên dưới). Dù chưa hội tụ, profile hành vi vẫn hợp lệ: di chuyển nhiều
nhất, `first_action_entropy ≈ 2.3` (chính sách còn rất bất định, đặc biệt ngay
sau transition), `goal_progress ≈ 0` (chưa học định hướng), `backtrack` cao.
PPO vượt random/tabular nhờ tổng quát hoá của NN, nhưng còn xa greedy.

**random** — Sàn dưới. Hiếm khi crossing gate (0.25/ep), không tiến bộ.

> **Ghi chú về PPO:** `train_map_to_map_ppo` sample task ngẫu nhiên toàn cục,
> bao gồm cả cặp start/goal khác component (bất khả thi) → success khi train ở
> mức ~1%. Checkpoint dùng ở đây vì vậy chưa hội tụ. Để có data point PPO mạnh,
> cần huấn luyện lại với task lọc theo component (tái dùng `solvable_tasks()`).

---

## 6. Kết luận & đề xuất

**Xếp hạng phục hồi sau transition:** `greedy` ≫ `ppo` > `tabular_q` ≈ `random`.

**Kết luận chính:**

1. Phục hồi định hướng tốt sau transition đòi hỏi **memory về cấu trúc map**
   (gate nào → zone nào) và **goal-conditioning**. Greedy có cả hai (zone
   graph) nên thắng tuyệt đối.
2. Dao động bám gate (`backtrack` cao) là bệnh lý transition phổ biến của các
   agent thiếu episodic memory — chúng quên gate vừa đi vào.
3. **Full observation không phải thuốc chữa bách bệnh**: nó không cải thiện
   greedy/ppo đáng kể, và làm tabular agent dao động tệ hơn.

**Đề xuất hướng tiếp theo:**

- Bổ sung **episodic memory về entry gate** cho agent học (one-hot gate vừa
  dùng, hoặc recurrent state) — kỳ vọng giảm `backtrack` mạnh.
- Huấn luyện lại PPO chỉ trên task cùng component để có data point deep-RL hội
  tụ; so sánh PPO recurrent vs feedforward về `backtrack_rate`.
- Thêm goal-conditioning vào state của tabular agent (ví dụ hướng BFS tới gate
  đúng) và đo lại — kiểm tra giả thuyết Q3.

**Artefact:**
- `results/transition_metrics.csv` — số liệu thô để vẽ biểu đồ.
- `state_sim/transition_benchmark.py` — chạy lại / mở rộng thực nghiệm.
- `state_sim/agent_interface.py` — thêm kiến trúc mới qua `AgentProtocol`.

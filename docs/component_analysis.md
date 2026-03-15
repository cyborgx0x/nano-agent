# Component Analysis — Hạn Chế & Hướng Thay Đổi

**Date:** 2026-03-01  
**Bối cảnh:** Agent gather trong simulated Albion Online — mục tiêu dạy **meta-skills** (exploration, planning, gather) generalizable sang zone chưa thấy.

**Vấn đề quan sát:**
- Training success dao động 33–68%, không hội tụ
- Holdout success = 0.00 (fail hoàn toàn trên zone chưa train)
- Episode length tăng 286 → 1169 (agent chậm dần)

---

## 1. Observation Encoding

**File:** [state_sim/ppo/encoder.py](../state_sim/ppo/encoder.py)

### Thiết kế hiện tại

```
obs_dim = 139
├── zone_id one-hot:   36 dims  (26%)   ← identity
├── goal_zone one-hot: 36 dims  (26%)   ← identity
├── zone_type:          5 dims  ( 4%)   ← property
├── biome:              5 dims  ( 4%)   ← property
├── task_type:          3 dims  ( 2%)
├── fsm_state:          8 dims  ( 6%)
├── continuous:        37 dims  (27%)
└── world_model:        9 dims  ( 6%)
```

### Vấn đề

**72/139 dims (52%) là zone IDENTITY, không phải zone PROPERTY.**

| Khía cạnh                                  | Phân tích                                                                                                                                                                         |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Gốc rễ**                                 | One-hot zone_id nói "tôi đang ở forest_gather_1" — không nói "zone này có gì". Mỗi zone = 1 neuron riêng biệt, không chia sẻ representation                                       |
| **Hệ quả**                                 | Network memorize "zone_id=4 → quay phải tìm resource" thay vì "thấy resource gần → đi tới". Khi gặp zone_id chưa train, gradient = 0 cho neuron đó → output random                |
| **Tại sao zone_type+biome không cứu được** | 10 dims (type+biome) vs 72 dims (zone_id). Gradient descent tự nhiên ưu tiên dùng 72 dims vì capacity cao hơn — network **có thể** dùng zone_type nhưng **không bị ép** phải dùng |
| **Bằng chứng**                             | `forest_gather_1` (holdout, idx=4, blue/forest) có zone_type/biome giống `forest_gather_2` (train, idx=5, yellow/forest) nhưng agent hoàn toàn fail trên holdout                  |

### Hướng thay đổi

**Loại bỏ zone_id one-hot, giữ zone properties:**

```python
# TRƯỚC (139 dims, 52% identity)
zone_id:   36 dims  # one-hot per zone — memorizable
goal_zone: 36 dims  # one-hot per zone — memorizable

# SAU (67 dims, 0% identity)
zone_type:  5 dims   # city/blue/yellow/red/black — generalizable
biome:      5 dims   # forest/highland/mountain/steppe/swamp — generalizable
zone_risk:  1 dim    # 0.0–1.0 continuous risk level
zone_tier:  1 dim    # normalized tier from zone_type
# goal encoding nếu cần navigation:
goal_distance:     1 dim    # đã có
goal_zone_type:    5 dims   # thay thế goal one-hot
goal_biome:        5 dims   # thay thế goal one-hot
```

**Lý do:** Agent phải học "trong zone blue/forest, resource ở đâu thì interact" — skill này transfer tự động sang mọi zone blue/forest, kể cả holdout.

**Rủi ro:** Mất khả năng phân biệt hai zone cùng type/biome (ví dụ forest_gather_1 vs forest_gather_2 đều là blue/forest). Nhưng đây chính là điều ta muốn — agent không nên cần biết zone nào, chỉ cần biết zone **như thế nào**.

---

## 2. Network Architecture

**File:** [state_sim/ppo/network.py](../state_sim/ppo/network.py)

### Thiết kế hiện tại

```
ActorCritic (feedforward, ~1.3M params)
├── LayerNorm(obs_dim=139)
├── Stem: Linear(139→512) → GELU → Linear(512→512) → GELU
├── MoE: 4 experts × Expert(512→768→512)
│   └── Gate: Linear(512→4) → softmax
├── Residual: 3 × ResidualBlock(512, dropout=0.1)
├── Policy Tower: LN → Linear(512→512) → GELU → Linear(512→29)
└── Value Tower: LN → Linear(512→512) → GELU → Linear(512→1)
```

### Vấn đề

| Khía cạnh                    | Phân tích                                                                                                                                                                                                                                                                                                                         |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Overcapacity**             | 1.3M params cho 139-dim input. Training data ~150K transitions (300ep × 500steps). Ratio params:data ≈ 1:115. Network thừa capacity để memorize mọi pattern thay vì generalize                                                                                                                                                    |
| **MoE Gate bị zone-coupled** | `expert_gate = Linear(512→4)` routing dựa trên stem output. Stem output bị dominated bởi zone_id (52%) → experts chuyên hóa theo zone thay vì theo skill. Expert 1 = "chuyên forest_gather_2", Expert 3 = "chuyên steppe_gather_1" — thay vì Expert 1 = "exploration", Expert 3 = "gather approach"                               |
| **No recurrence**            | Feedforward-only: agent không giữ temporal context qua steps. World model bổ sung 9 features nhưng chỉ là summary statistics (explored_ratio, resource_count). Agent không biết trình tự hành động ("vừa gather xong → nên tìm resource tiếp" vs "đang explore → nên chuyển gathering") — phải suy ra hoàn toàn từ state hiện tại |
| **Dropout = 0.1**            | Quá thấp cho model overcapacity. Không đủ regularization để chống memorization                                                                                                                                                                                                                                                    |

### Hướng thay đổi

**Phương án A — Giảm capacity + tăng regularization:**

```python
# Giảm hidden_dim
hidden_dim = 256        # từ 512
expert_hidden = 384     # từ 768
n_experts = 2           # từ 4
dropout = 0.2           # từ 0.1
# ~200K params — buộc network phải generalize
```

**Phương án B — Thêm skill-based MoE gating (nếu giữ MoE):**

```python
# Gate dựa trên task features thay vì full stem output
task_features = concat(zone_type, biome, fsm_state, task_type)  # 21 dims
self.expert_gate = nn.Linear(21, n_experts)  # Gate theo skill context
```

**Phương án C — Thêm lightweight recurrence:**

```python
# GRU nhỏ để giữ temporal context
self.gru = nn.GRU(hidden_dim, 128, batch_first=True)
# Giúp agent nhớ "vừa gather → tìm resource tiếp" vs "đang walk → explore"
```

**Khuyến nghị:** Phương án A (giảm capacity) kết hợp với fix encoding ở mục 1 — đơn giản nhất, ít rủi ro.

---

## 3. Reward Structure

**File:** [state_sim/environment.py](../state_sim/environment.py) (dòng 940–1050), [state_sim/curriculum.py](../state_sim/curriculum.py)

### Thiết kế hiện tại — Phase 5 (gather mode)

```python
reward = (
    resource_gain * 1.2 / 100.0          # ~0.012 per gather event
    + xp_gain * 0.1 / 100.0              # ~0.001
    + bank_gain * 1.4 / 100.0            # ~0.014 per bank
    - damage_taken * 0.95                 # -0.95 per hit
    - idle_penalty * 0.02                 # -0.02 per idle tick
    - 0.003                               # step penalty (mỗi step)
    - 12.0 if died                        # death penalty
    + exploration_reward                   # 0.04 per new cell
    - stagnation_pen                      # -0.015 * multiplier
    - no_move_pen                          # -0.008 if no move
    + gather_bonus                         # 0.06–0.14 per gather
    + resource_approach_reward             # 0.0–0.02 per step closer
)
+ 3.0 if inventory_full                   # bonus khi đầy 99 items
```

### Vấn đề

| Khía cạnh                                   | Phân tích                                                                                                                                                                                                                                       |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Reward scale không cân**                  | `resource_gain * 1.2 / 100.0 = 0.012` — per-gather reward rất nhỏ. Mỗi step bị trừ 0.003. Agent cần 4 steps di chuyển tới resource, sau đó 3 ticks channel → net reward ≈ 0.012 + 0.06 - 7×0.003 = 0.051. Marginal benefit rất thấp             |
| **Step penalty chống lại long episodes**    | -0.003/step × 1500 steps = -4.5 penalty tối đa. Nhưng fill inventory cần ~300–600 steps → agent bị phạt -0.9 đến -1.8 chỉ vì bước. Kết hợp với reward per gather thấp → incentive chính là SỐNG (tránh death penalty -12) chứ không phải GATHER |
| **Gather_bonus scaling ngược**              | `0.06 + 0.08 * progress` — bonus TĂNG khi gần đầy inventory. Nhưng khó nhất là KHỞI ĐẦU gather (khi chưa biết resource ở đâu). Nên đảo lại: bonus lớn ở đầu (encourage bắt đầu gather), giảm khi đã ổn định                                     |
| **Inventory-full threshold = 99**           | Binary success critera: gather 98 = fail (success=0.59), gather 99 = pass (success=0.70+). Trên zone khó (red/black, ít resource, nhiều mob), 99 gần bất khả thi trong 1500 steps. Tạo cliff effect trong gradient signal                       |
| **Exploration bonus cạnh tranh với gather** | +0.04 per new cell vs +0.06 per gather event. Exploration được thưởng gần bằng gathering → agent có thể "tối ưu" bằng cách đi quanh map thay vì gather. Giải thích episode length tăng 286→1169: agent học explore thay vì gather               |
| **Resource approach shaping yếu**           | Max 0.02 per step, chỉ khi di chuyển >0.005 gần resource hơn. Bị gate bởi `resource_gain == 0` — tốt, nhưng không đủ mạnh để dẫn agent qua zone lớn tới resource xa                                                                             |

### Hướng thay đổi

**A. Tăng gather incentive, giảm exploration incentive:**

```python
# Gather mạnh hơn
gather_bonus = 0.15 + 0.10 * (1 - progress)  # Lớn lúc đầu, giảm dần
resource_approach_reward = min(0.05, delta * 0.3)  # Mạnh hơn

# Exploration yếu hơn trong gather mode
if task_type == "gather":
    exploration_bonus = 0.01   # từ 0.04 — chỉ khuyến khích nhẹ
```

**B. Giảm step penalty hoặc scale theo task:**

```python
# Step penalty thấp hơn cho gather (cần thời gian)
step_penalty = 0.001 if task_type == "gather" else 0.003
```

**C. Thay đổi success metric (xem mục 5).**

---

## 4. Training Strategy

**File:** [state_sim/ppo/trainer.py](../state_sim/ppo/trainer.py), [state_sim/ppo/config.py](../state_sim/ppo/config.py)

### Thiết kế hiện tại

```
PPO Config:
├── episodes: 300
├── max_steps: 1500
├── lr: 1e-4
├── clip_eps: 0.15
├── update_every_episodes: 10
├── minibatch_size: 256
├── Teacher forcing (gather):
│   ├── start: 0.80 (80% teacher actions)
│   ├── min: 0.35 (never below 35%)
│   ├── decay: power=2.0 (quadratic decay)
│   └── recovery: boost=0.20, ratio=0.75
├── Holdout:
│   ├── ratio: 0.2 (6/31 zones held out)
│   ├── eval_interval: 30 episodes
│   └── eval_episodes: 10
└── Zone scheduler: round_robin (25 train zones)
```

### Vấn đề

| Khía cạnh                                                    | Phân tích                                                                                                                                                                                                                                                                                                                                             |
| ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Teacher min quá cao**                                      | `gather_teacher_min = 0.35` → 35% actions LUÔN là teacher ở cuối training. Agent không bao giờ fully autonomous. Khi holdout eval (0% teacher) → performance cliff. Training success bao gồm 35% "free correct actions" từ teacher — overstates actual agent ability                                                                                  |
| **Teacher action là optimal, agent action không được train** | Khi teacher fires, `logprob = dist.log_prob(teacher_action)` — gradient cập nhật policy TOWARD teacher action. Nhưng teacher biết exact resource positions (`env.zone_entities`), còn agent chỉ có vision. Agent được dạy "go to resource at (0.3, 0.7)" nhưng obs chỉ cho thấy `nearest_resource_dist=1.0` (ngoài vision). Learning signal mâu thuẫn |
| **300 episodes không đủ**                                    | 25 train zones × round_robin = mỗi zone ~12 episodes. Agent chưa đủ thời gian học 1 zone thì đã chuyển sang zone tiếp theo. Minimum nên là 50 episodes/zone = 1250 episodes                                                                                                                                                                           |
| **Holdout eval dùng argmax**                                 | `action = torch.argmax(logits)` thay vì sampling. Trên zone chưa train, logits bị noise → argmax lock vào 1 action sai lặp đi lặp lại. Đây là lý do holdout = 0.00 **tuyệt đối** thay vì chỉ thấp                                                                                                                                                     |
| **Eval interval misaligned**                                 | `gather_eval_interval=30` nhưng `log_interval=20`. Chỉ ở episodes chia hết cho cả 30 VÀ 20 (tức 60, 120, 180, 240, 300) holdout mới chạy. Các log khác in `holdout=0.00` dù đó chỉ là default value, dễ gây hiểu nhầm                                                                                                                                 |
| **Checkpoint score mix holdout**                             | `checkpoint_score = success * 0.55 + holdout * 0.45`. Khi holdout = 0, checkpoint chỉ dựa trên 55% train success → model overfitting vẫn được save                                                                                                                                                                                                    |
| **Round-robin không shuffle**                                | Agent thấy zone theo thứ tự cố định → có thể memorize sequence pattern thay vì zone properties                                                                                                                                                                                                                                                        |

### Hướng thay đổi

**A. Teacher forcing decay hoàn toàn:**

```python
gather_teacher_start = 0.80
gather_teacher_min = 0.05    # từ 0.35 → gần 0
gather_teacher_decay_power = 1.5  # linear-ish thay vì quadratic
```

**B. Tăng episodes hoặc giảm zone count:**

```python
episodes = 1500              # từ 300 → đủ ~60 ep/zone
# HOẶC: bắt đầu với 5-8 zones, tăng dần
```

**C. Holdout eval dùng sampling thay vì argmax:**

```python
# Trong _evaluate_gather_holdout:
dist = Categorical(logits=logits)
action = int(dist.sample().item())  # thay vì argmax
```

**D. Align eval với log interval:**

```python
gather_eval_interval = 20    # = log_interval → print holdout mỗi lần log
```

---

## 5. Evaluation Metric (gather_success)

**File:** [state_sim/environment.py](../state_sim/environment.py) (dòng 1084–1105)

### Thiết kế hiện tại

```python
if inventory_full (>= 99 events):
    gather_success = 0.7 + 0.3 * efficiency    # 0.70 – 1.00
else:
    gather_success = 0.6 * (events / 99)        # 0.00 – 0.59
```

```
Score visualization:
0 events ─────────────── 98 events ── 99 events ──── 99 (fast)
   0.0                     0.59    │    0.70            1.0
                                   │
                              CLIFF (+0.11 jump)
```

### Vấn đề

| Khía cạnh                                   | Phân tích                                                                                                                                                                                 |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Binary cliff tại 99**                     | Gather 98 items = success 0.59. Gather 99 items = success 0.70+. Nhảy +0.11 chỉ vì 1 event. Gradient signal bị discontinuous — PPO khó optimize qua cliff                                 |
| **Partial credit bị cap tại 0.6**           | Dù gather 90/99 items (gần hoàn thành), success chỉ 0.545. Agent đã làm tốt 91% nhưng chỉ nhận 55% credit                                                                                 |
| **Efficiency bonus chỉ cho full inventory** | Gather 99 items nhanh = bonus. Nhưng gather 80 items nhanh = không bonus. Không khuyến khích hiệu quả ở partial completion                                                                |
| **99 events quá cứng cho zone khó**         | Red/black zones có nhiều mob, ít resource, cần dodge. 99 events trong 1500 steps yêu cầu ~1 event per 15 steps — rất tight. Metric penalize agent vì zone difficulty, không phải vì skill |
| **Không phân biệt quality**                 | Gather 99 tier-1 resources = Gather 99 tier-4 resources. Nhưng tier-4 có giá trị cao hơn nhiều. Agent không được khuyến khích tìm resource tốt hơn                                        |

### Hướng thay đổi

**Phương án Khuyến nghị — Continuous ratio + efficiency bonus:**

```python
def gather_success_v2(gather_events, max_steps, step_count, inventory_target):
    # Continuous: mỗi event đóng góp đều
    gather_ratio = min(1.0, gather_events / inventory_target)

    # Efficiency bonus khi gather > 50% (khuyến khích nhanh)
    speed_bonus = 0.0
    if gather_ratio > 0.5:
        efficiency = max(0.0, 1.0 - step_count / max_steps)
        speed_bonus = efficiency * 0.15

    # Completion bonus nhẹ (không cliff mà gradual)
    completion_bonus = 0.0
    if gather_ratio >= 0.9:
        completion_bonus = (gather_ratio - 0.9) * 1.0  # 0 → 0.1

    return gather_ratio * 0.75 + speed_bonus + completion_bonus
```

**So sánh hai metric:**

| Gather events | Metric cũ | Metric mới (v2)           | Đánh giá                  |
| ------------- | --------- | ------------------------- | ------------------------- |
| 0             | 0.00      | 0.00                      | Giống                     |
| 30            | 0.18      | 0.23                      | v2 công bằng hơn          |
| 60            | 0.36      | 0.46                      | v2 reward partial tốt hơn |
| 90            | 0.55      | 0.68 + speed              | v2 ghi nhận gần-complete  |
| 98            | 0.59      | 0.74 + speed + completion | v2 smooth, không cliff    |
| 99            | 0.70+     | 0.75 + speed + completion | v2 tương đương, smooth    |

---

## 6. Holdout Evaluation

**File:** [state_sim/ppo/trainer.py](../state_sim/ppo/trainer.py) (dòng 42–100, 603–640)

### Thiết kế hiện tại

```python
def _evaluate_gather_holdout(...):
    eval_model = ActorCritic(obs_dim, action_dim, memory_size)
    eval_model.load_state_dict(model.state_dict())
    eval_model.eval()

    for ep in range(episodes):
        zone_id = zone_cycle[ep % len(zone_cycle)]
        env.force_next_gather_spawn_zone(zone_id)
        obs = env.reset()
        while not done:
            obs_t = encode(obs)
            logits, _, hidden = eval_model(obs_t, hidden)
            action = int(torch.argmax(logits, dim=-1))  # ← DETERMINISTIC
            obs, _, done, info = env.step(action)
        successes.append(info["gather_success"])
```

### Vấn đề

| Khía cạnh                        | Phân tích                                                                                                                                                                               |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Argmax trên noisy logits**     | Zone chưa train → zone_id one-hot neuron chưa có weight → logits gần uniform random. Argmax chọn 1 action cố định mỗi state → agent lặp đi lặp lại cùng action → stuck, không gather gì |
| **Không có teacher fallback**    | Training có 35% teacher minimum, nhưng eval = 0% teacher. Gap quá lớn: agent "tốt" trong training vì teacher giúp, fail hoàn toàn trong eval                                            |
| **Chỉ 10 episodes per eval**     | Với 6 holdout zones, mỗi zone chỉ ~1.7 episodes. Variance quá cao cho signal ổn định                                                                                                    |
| **Eval ở episode 300 mới chạy**  | `eval_interval=30`, trong 300 episodes chỉ run holdout 10 lần. Không đủ feedback sớm để thấy holdout fail                                                                               |
| **Không log per-zone breakdown** | Chỉ in mean holdout success. Không biết zone nào fail —  có thể 4/6 zone pass nhưng 2 zone catastrophic fail kéo mean xuống 0                                                           |

### Hướng thay đổi

**A. Sampling thay vì argmax:**

```python
dist = Categorical(logits=logits)
action = int(dist.sample().item())
```

**B. Eval interval = log interval:**

```python
gather_eval_interval = log_interval  # luôn print holdout cùng lúc
```

**C. Tăng eval episodes + log per-zone:**

```python
gather_eval_episodes = 30  # 5 per holdout zone
# Print per-zone breakdown
for zone in holdout_zones:
    print(f"  holdout {zone}: {zone_success:.2f}")
```

---

## 7. Teacher Forcing

**File:** [state_sim/ppo/trainer.py](../state_sim/ppo/trainer.py) (dòng 153–240)

### Thiết kế hiện tại

Teacher gather function (`_teacher_action_gather`) biết:
- Toàn bộ `env.zone_entities` (exact positions)
- Zone graph (navigate to city)
- Nearest resource bằng omniscient distance

Agent chỉ biết:
- Entities trong vision radius (0.30)
- Memory features (decayed)
- World model summary (9 dims)

### Vấn đề

| Khía cạnh                              | Phân tích                                                                                                                                                                                                                                                              |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Information asymmetry**              | Teacher biết resource tại (0.7, 0.3) nhưng agent obs chỉ thấy `nearest_resource_dist=1.0` (ngoài vision). Agent được train gradient hướng tới action "đi tới (0.7, 0.3)" nhưng không có thông tin để tái tạo quyết định đó. Learning signal mâu thuẫn → gradient noise |
| **Teacher quá giỏi**                   | Teacher always goes to nearest resource. Agent chỉ cần copy teacher → không cần học exploration hay planning. Khi teacher giảm, agent thiếu skill tự lập                                                                                                               |
| **Min floor 35% quá cao**              | Cuối training vẫn 35% teacher → agent LUÔN bị "giúp". Success rate bao gồm teacher contribution → overestimates agent ability. Holdout (0% teacher) → catastrophic drop                                                                                                |
| **Recovery mechanism tạo oscillation** | Khi success drops dưới 75% of best → teacher boost +20%. Boost decays 4%/episode. Cycle: agent improves → teacher decreases → agent drops → teacher increases → agent improves again. Không hội tụ                                                                     |

### Hướng thay đổi

**A. Vision-consistent teacher:**

```python
def _teacher_action_gather_v2(env):
    # Chỉ dùng entities TRONG vision radius, giống agent
    visible_resources = [r for r in resources if dist(agent, r) < 0.30]
    if visible_resources:
        go_to_nearest(visible_resources)
    else:
        explore_randomly()  # giống agent khi không thấy
```

**B. Decay teacher tới gần 0:**

```python
gather_teacher_min = 0.05    # từ 0.35
gather_teacher_decay_power = 1.5
# Sau 70% training, teacher < 10%
```

**C. Gradual curriculum thay vì sudden eval:**

```python
# Eval cũng có low teacher probability
eval_teacher_prob = 0.10  # nhẹ, để agent không shock
# Giảm eval_teacher_prob dần tới 0
```

---

## Tổng Hợp Ưu Tiên

### Phải fix (gốc rễ overfitting):

| Thứ tự | Thay đổi                       | Tác động                                                  |
| ------ | ------------------------------ | --------------------------------------------------------- |
| 1      | **Loại bỏ zone_id one-hot**    | Buộc agent học properties thay vì identity. Fix holdout=0 |
| 2      | **Giảm teacher_min → 0.05**    | Agent phải tự lập, success phản ánh thực lực              |
| 3      | **Holdout eval dùng sampling** | Cho agent cơ hội explore trên zone mới                    |

### Nên fix (cải thiện convergence):

| Thứ tự | Thay đổi                                       | Tác động                            |
| ------ | ---------------------------------------------- | ----------------------------------- |
| 4      | **Giảm network capacity**                      | Chống memorization, buộc generalize |
| 5      | **Tăng gather reward, giảm exploration bonus** | Agent focus gather thay vì wander   |
| 6      | **Continuous success metric**                  | Gradient signal smooth, không cliff |

### Tùy chọn (nice-to-have):

| Thứ tự | Thay đổi                               | Tác động                        |
| ------ | -------------------------------------- | ------------------------------- |
| 7      | **Vision-consistent teacher**          | Learning signal không mâu thuẫn |
| 8      | **Tăng episodes hoặc zone curriculum** | Đủ experience per zone          |
| 9      | **Align eval/log intervals**           | Debug dễ hơn                    |

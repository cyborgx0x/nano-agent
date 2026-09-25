## Unified problem hypothesis (MVP)

Thay vì chia tách độc lập, gộp thành 1 objective chung:

**Survival-Economy Loop**

- Farm tài nguyên (map xanh + vàng)
- Né mob / giảm rủi ro chết
- Farm mob mức nhẹ để lấy XP
- Quay về thành và cất tài nguyên vào rương khi đủ điều kiện

## Scope MVP

- Chỉ chạy ở map xanh + vàng
- Chưa vào map đỏ / map đen
- Chưa cần perception hoàn chỉnh từ game thật
- Ưu tiên train trong state-based simulator trước

## Curriculum (fine-grained)

### Phase 0 - Safety bootstrap

- Giữ trạng thái an toàn cơ bản
- Tránh vào vùng nguy hiểm giả lập

### Phase 1 - In-zone navigation

- Di chuyển từ điểm A -> B trong cùng map
- Không có mob, không gather

### Phase 2 - Inter-zone transition

- Qua cổng giữa 2 map và quay lại map gốc

### Phase 3 - Single gather loop

- Gather 1 node gần nhất
- Quay về safe point

### Phase 4 - Gather with threat

- Gather khi có mob đơn giản
- Né mob trong bán kính nguy hiểm

### Phase 5 - Economy loop

- Tối ưu vòng gather -> bank với inventory giới hạn

### Phase 6 - XP integration

- Thêm farm mob nhẹ để lấy XP
- Vẫn ưu tiên sống sót và giữ tài nguyên

### Phase 7 - Mixed random scenarios

- Trộn objective resource + XP + return-to-city
- Spawn ngẫu nhiên tài nguyên/mob để tăng độ robust

## Promotion criteria (lên phase)

- `success_rate >= 0.85`
- `death_rate <= 0.05`
- `avg_return_time` dưới ngưỡng từng phase
- Giữ 10-20% episode từ phase cũ để chống quên

## State / Action / Reward (state-based sim)

### State (tối thiểu)

- Zone hiện tại, loại zone (blue/yellow)
- Vị trí tương đối trong zone
- HP, trạng thái mount
- Inventory load, resource value đang giữ
- Khoảng cách về city/portal gần nhất
- Threat score (mob density x proximity)

### Action (high-level)

- MoveToPOI
- MoveToGate
- GatherResource
- FarmMob
- BankAtCity
- RetreatSafe
- MountToggle

### Reward (dense + shaped)

- Dương: giá trị tài nguyên thu được, XP thu được, hoàn thành bank thành công
- Âm: chết, HP thấp kéo dài, idle quá lâu, đi vòng/di chuyển không hiệu quả

## Success definition

Trong sim, agent đạt policy ổn định nếu:

- Duy trì lợi nhuận dương theo episode
- Tỉ lệ chết thấp bền vững
- Hoàn thành đều vòng gather/farm -> return -> bank

Khi đạt, mới chuyển sang bước mapping perception + control vào game thật.

# TAL VLM Review Combined Analysis

## Summary

| Metric | Value |
|---|---:|
| reviewed_total | 15 |
| gt_hit_by_tal_only_reviewed | 5 |
| tal_only_non_gt_reviewed | 10 |
| vlm_confirmed_total | 3 |
| vlm_rejected_total | 12 |
| vlm_confirmed_gt_added | 2 |
| vlm_confirmed_non_gt | 1 |
| vlm_rejected_non_gt | 9 |
| overall_rejection_rate | 0.8000 |
| non_gt_rejection_rate | 0.9000 |
| non_gt_confirmation_rate | 0.1000 |

## Confirmed Proposals

| Group | Clip | Action | Score | Confidence | Evidence |
|---|---|---|---:|---:|---|
| gt_hit_by_tal_only | `1Y09V_c109_t0.00_30.79` | holding_drink_container | 0.0444 | 0.95 | The person is holding a white cup on a saucer in multiple frames, including frame 0, 1, 2, 3, 4, and 5. |
| gt_hit_by_tal_only | `38TF8_c106_t0.00_31.50` | holding_drink_container | 0.0436 | 0.95 | The man in the background is holding a white cup in his hand across multiple frames (0-5). |
| tal_only_non_gt | `L57L2_c125_t22.10_123.70` | standing_up | 0.0521 | 0.95 | Person is seen transitioning from a crouched/sitting position on a box to standing upright across frames 0 to 5. |

## Rejected Non-GT Proposals

| Clip | Action | Score | Evidence |
|---|---|---:|---|
| `SPG5Q_c150_t9.20_33.42` | sitting_on_chair | 0.0483 | The person is standing and bending over, not seated on a chair. |
| `SPG5Q_c150_t9.20_33.42` | standing_up | 0.0448 | Person is already standing and adjusting clothing, no evidence of transitioning from sitting or crouching. |
| `1Y09V_c109_t0.00_30.79` | standing_up | 0.0422 | Person is already standing and holding a cup, no transition from sitting or crouching is visible. |
| `YVH4J_c051_t0.00_38.30` | sitting_down | 0.0388 | The person is already seated on the bed and appears to be reading a book throughout the frames. |
| `SPG5Q_c150_t9.20_33.42` | putting_drink_container | 0.0379 | No drink container is visible being placed down by the person. An orange bottle is visible on a shelf in the background but not being interacted with. |
| `YVH4J_c051_t0.00_38.30` | sitting_on_chair | 0.0372 | Person is lying on stomach on a bed, not seated on a chair. |
| `QU2WL_c019_t0.00_34.21` | standing_up | 0.0369 | The person is standing in frames 0-4 and then sits down in frame 5, which is a sitting action, not standing up. |
| `1Y09V_c109_t0.00_30.79` | sitting_down | 0.0363 |  |
| `7WIKW_c151_t17.10_42.10` | pouring_drink_container | 0.0363 |  |

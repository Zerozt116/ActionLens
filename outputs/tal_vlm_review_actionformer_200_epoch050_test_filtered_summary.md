# TAL Proposal VLM Review

## Summary

| Metric | Value |
|---|---:|
| selection_outcomes | ['gt_hit_by_tal_only', 'tal_only_non_gt'] |
| filter_top_k_per_clip | 3 |
| filter_min_score | 0.04 |
| filter_min_duration | 0.5 |
| filter_max_duration | 30.0 |
| limit | 999 |
| candidates_before_filter | 85 |
| candidates_after_filter | 16 |
| selected_for_review | 16 |
| filtered_total | 69 |
| filtered_by_rank | 55 |
| filtered_by_score | 0 |
| filtered_by_duration | 33 |
| proposals_selected | 16 |
| completed | 16 |
| failed | 0 |
| vlm_confirmed | 4 |
| vlm_weak_confirmed | 0 |
| vlm_rejected | 12 |
| dry_run | 0 |

## Reviews

| Clip | Action | Outcome | TAL score | TAL IoU | VLM decision | Confidence |
|---|---|---|---:|---:|---|---:|
| `5B8M5_c156_t0.00_30.71` | sitting_on_chair | gt_hit_by_tal_only | 0.2660 | 0.926 | vlm_confirmed | 0.95 |
| `K8AUX_c110_t0.00_29.54` | holding_drink_container | gt_hit_by_tal_only | 0.2435 | 0.889 | vlm_confirmed | 0.95 |
| `8BG1T_c106_t5.60_45.46` | sitting_down | gt_hit_by_tal_only | 0.3211 | 0.729 | vlm_confirmed | 0.95 |
| `K8AUX_c110_t0.00_29.54` | walking_through_doorway | gt_hit_by_tal_only | 0.1127 | 0.666 | vlm_rejected | 0.00 |
| `1BUFQ_c132_t0.00_40.25` | sitting_on_chair | tal_only_non_gt | 0.3467 | 0.000 | vlm_rejected | 0.00 |
| `1BUFQ_c132_t0.00_40.25` | holding_phone | tal_only_non_gt | 0.2392 | 0.000 | vlm_rejected | 0.00 |
| `406LH_c097_t1.00_21.00` | sitting_down | tal_only_non_gt | 0.2195 | 0.000 | vlm_rejected | 0.00 |
| `406LH_c097_t1.00_21.00` | holding_drink_container | tal_only_non_gt | 0.1992 | 0.000 | vlm_confirmed | 0.95 |
| `406LH_c097_t1.00_21.00` | pouring_drink_container | tal_only_non_gt | 0.1759 | 0.000 | vlm_rejected | 0.00 |
| `SPG5Q_c150_t9.20_33.42` | drinking_water | tal_only_non_gt | 0.1576 | 0.000 | vlm_rejected | 0.00 |
| `SPG5Q_c150_t9.20_33.42` | holding_drink_container | tal_only_non_gt | 0.1568 | 0.000 | vlm_rejected | 0.00 |
| `5B8M5_c156_t0.00_30.71` | looking_at_phone | tal_only_non_gt | 0.1458 | 0.000 | vlm_rejected | 0.00 |
| `S1BYH_c150_t0.00_28.04` | holding_drink_container | tal_only_non_gt | 0.1319 | 0.000 | vlm_rejected | 0.00 |
| `5B8M5_c156_t0.00_30.71` | holding_phone | tal_only_non_gt | 0.1304 | 0.000 | vlm_rejected | 0.00 |
| `S1BYH_c150_t0.00_28.04` | taking_phone | tal_only_non_gt | 0.0974 | 0.000 | vlm_rejected | 0.00 |
| `S1BYH_c150_t0.00_28.04` | walking_through_doorway | tal_only_non_gt | 0.0970 | 0.000 | vlm_rejected | 0.00 |

## Evidence

### `5B8M5_c156_t0.00_30.71` / sitting_on_chair

- Window: 0.00s - 29.56s
- Review dir: `outputs/tal_vlm_review_actionformer_200_epoch050_test_filtered/5B8M5_c156_t0.00_30.71/sitting_on_chair_0.00_29.56`
- Evidence: The person is visibly seated on a blue chair in frames 0, 1, 2, 4, and 5. In frame 3, the person is partially occluded but still appears to be seated.

### `K8AUX_c110_t0.00_29.54` / holding_drink_container

- Window: 0.00s - 27.36s
- Review dir: `outputs/tal_vlm_review_actionformer_200_epoch050_test_filtered/K8AUX_c110_t0.00_29.54/holding_drink_container_0.00_27.36`
- Evidence: The person is holding a blue cup in frames 0-2, actively drinking from it.

### `8BG1T_c106_t5.60_45.46` / sitting_down

- Window: 2.30s - 14.02s
- Review dir: `outputs/tal_vlm_review_actionformer_200_epoch050_test_filtered/8BG1T_c106_t5.60_45.46/sitting_down_2.30_14.02`
- Evidence: The person is seen standing at frame 0, then bending and lowering themselves onto the bench by frame 3, and is fully seated by frame 4.

### `K8AUX_c110_t0.00_29.54` / walking_through_doorway

- Window: 23.72s - 30.47s
- Review dir: `outputs/tal_vlm_review_actionformer_200_epoch050_test_filtered/K8AUX_c110_t0.00_29.54/walking_through_doorway_23.72_30.47`
- Evidence: The person is standing in the doorway holding items and does not appear to be moving through it. The person is stationary or moving very slowly within the doorway area.

### `1BUFQ_c132_t0.00_40.25` / sitting_on_chair

- Window: 14.72s - 26.40s
- Review dir: `outputs/tal_vlm_review_actionformer_200_epoch050_test_filtered/1BUFQ_c132_t0.00_40.25/sitting_on_chair_14.72_26.40`
- Evidence: The person is standing throughout the sampled frames.

### `1BUFQ_c132_t0.00_40.25` / holding_phone

- Window: 2.44s - 34.19s
- Review dir: `outputs/tal_vlm_review_actionformer_200_epoch050_test_filtered/1BUFQ_c132_t0.00_40.25/holding_phone_2.44_34.19`

### `406LH_c097_t1.00_21.00` / sitting_down

- Window: 0.54s - 10.61s
- Review dir: `outputs/tal_vlm_review_actionformer_200_epoch050_test_filtered/406LH_c097_t1.00_21.00/sitting_down_0.54_10.61`
- Evidence: Person is standing throughout the sampled frames, interacting with a door.

### `406LH_c097_t1.00_21.00` / holding_drink_container

- Window: 1.73s - 18.80s
- Review dir: `outputs/tal_vlm_review_actionformer_200_epoch050_test_filtered/406LH_c097_t1.00_21.00/holding_drink_container_1.73_18.80`
- Evidence: Person is holding a white cup or glass in hand, visible in frames 2-5.

### `406LH_c097_t1.00_21.00` / pouring_drink_container

- Window: 2.14s - 12.65s
- Review dir: `outputs/tal_vlm_review_actionformer_200_epoch050_test_filtered/406LH_c097_t1.00_21.00/pouring_drink_container_2.14_12.65`
- Evidence: No drink container or pouring action is visible. Person is holding a phone and opening a door.

### `SPG5Q_c150_t9.20_33.42` / drinking_water

- Window: 0.00s - 9.33s
- Review dir: `outputs/tal_vlm_review_actionformer_200_epoch050_test_filtered/SPG5Q_c150_t9.20_33.42/drinking_water_0.00_9.33`
- Evidence: The person is bending over and interacting with items on the counter, but no cup, bottle, or drinking action is visible.

### `SPG5Q_c150_t9.20_33.42` / holding_drink_container

- Window: 0.00s - 25.22s
- Review dir: `outputs/tal_vlm_review_actionformer_200_epoch050_test_filtered/SPG5Q_c150_t9.20_33.42/holding_drink_container_0.00_25.22`

### `5B8M5_c156_t0.00_30.71` / looking_at_phone

- Window: 0.00s - 30.22s
- Review dir: `outputs/tal_vlm_review_actionformer_200_epoch050_test_filtered/5B8M5_c156_t0.00_30.71/looking_at_phone_0.00_30.22`
- Evidence: No phone is visible in any of the sampled frames.

### `S1BYH_c150_t0.00_28.04` / holding_drink_container

- Window: 0.00s - 22.98s
- Review dir: `outputs/tal_vlm_review_actionformer_200_epoch050_test_filtered/S1BYH_c150_t0.00_28.04/holding_drink_container_0.00_22.98`

### `5B8M5_c156_t0.00_30.71` / holding_phone

- Window: 0.00s - 30.22s
- Review dir: `outputs/tal_vlm_review_actionformer_200_epoch050_test_filtered/5B8M5_c156_t0.00_30.71/holding_phone_0.00_30.22`

### `S1BYH_c150_t0.00_28.04` / taking_phone

- Window: 0.00s - 15.60s
- Review dir: `outputs/tal_vlm_review_actionformer_200_epoch050_test_filtered/S1BYH_c150_t0.00_28.04/taking_phone_0.00_15.60`

### `S1BYH_c150_t0.00_28.04` / walking_through_doorway

- Window: 0.00s - 21.64s
- Review dir: `outputs/tal_vlm_review_actionformer_200_epoch050_test_filtered/S1BYH_c150_t0.00_28.04/walking_through_doorway_0.00_21.64`
- Evidence: No doorway is visible in any of the sampled frames. The person is moving within a cluttered garage space, but there is no evidence of a doorway or crossing one.

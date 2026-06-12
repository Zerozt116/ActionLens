# TAL Proposal VLM Review

## Summary

| Metric | Value |
|---|---:|
| proposals_selected | 10 |
| completed | 10 |
| failed | 0 |
| vlm_confirmed | 1 |
| vlm_weak_confirmed | 0 |
| vlm_rejected | 9 |
| dry_run | 0 |

## Reviews

| Clip | Action | Outcome | TAL score | TAL IoU | VLM decision | Confidence |
|---|---|---|---:|---:|---|---:|
| `L57L2_c125_t22.10_123.70` | standing_up | tal_only_non_gt | 0.0521 | 0.000 | vlm_confirmed | 0.95 |
| `SPG5Q_c150_t9.20_33.42` | sitting_on_chair | tal_only_non_gt | 0.0483 | 0.000 | vlm_rejected | 0.00 |
| `SPG5Q_c150_t9.20_33.42` | standing_up | tal_only_non_gt | 0.0448 | 0.000 | vlm_rejected | 0.00 |
| `1Y09V_c109_t0.00_30.79` | standing_up | tal_only_non_gt | 0.0422 | 0.000 | vlm_rejected | 0.00 |
| `YVH4J_c051_t0.00_38.30` | sitting_down | tal_only_non_gt | 0.0388 | 0.000 | vlm_rejected | 0.00 |
| `SPG5Q_c150_t9.20_33.42` | putting_drink_container | tal_only_non_gt | 0.0379 | 0.000 | vlm_rejected | 0.00 |
| `YVH4J_c051_t0.00_38.30` | sitting_on_chair | tal_only_non_gt | 0.0372 | 0.000 | vlm_rejected | 0.00 |
| `QU2WL_c019_t0.00_34.21` | standing_up | tal_only_non_gt | 0.0369 | 0.000 | vlm_rejected | 0.00 |
| `1Y09V_c109_t0.00_30.79` | sitting_down | tal_only_non_gt | 0.0363 | 0.000 | vlm_rejected | 0.00 |
| `7WIKW_c151_t17.10_42.10` | pouring_drink_container | tal_only_non_gt | 0.0363 | 0.000 | vlm_rejected | 0.00 |

## Evidence

### `L57L2_c125_t22.10_123.70` / standing_up

- Window: 99.18s - 102.60s
- Review dir: `outputs/tal_vlm_review_epoch034_non_gt/L57L2_c125_t22.10_123.70/standing_up_99.18_102.60`
- Evidence: Person is seen transitioning from a crouched/sitting position on a box to standing upright across frames 0 to 5.

### `SPG5Q_c150_t9.20_33.42` / sitting_on_chair

- Window: 0.00s - 5.97s
- Review dir: `outputs/tal_vlm_review_epoch034_non_gt/SPG5Q_c150_t9.20_33.42/sitting_on_chair_0.00_5.97`
- Evidence: The person is standing and bending over, not seated on a chair.

### `SPG5Q_c150_t9.20_33.42` / standing_up

- Window: 9.34s - 23.78s
- Review dir: `outputs/tal_vlm_review_epoch034_non_gt/SPG5Q_c150_t9.20_33.42/standing_up_9.34_23.78`
- Evidence: Person is already standing and adjusting clothing, no evidence of transitioning from sitting or crouching.

### `1Y09V_c109_t0.00_30.79` / standing_up

- Window: 1.88s - 5.03s
- Review dir: `outputs/tal_vlm_review_epoch034_non_gt/1Y09V_c109_t0.00_30.79/standing_up_1.88_5.03`
- Evidence: Person is already standing and holding a cup, no transition from sitting or crouching is visible.

### `YVH4J_c051_t0.00_38.30` / sitting_down

- Window: 37.24s - 39.30s
- Review dir: `outputs/tal_vlm_review_epoch034_non_gt/YVH4J_c051_t0.00_38.30/sitting_down_37.24_39.30`
- Evidence: The person is already seated on the bed and appears to be reading a book throughout the frames.

### `SPG5Q_c150_t9.20_33.42` / putting_drink_container

- Window: 9.88s - 15.24s
- Review dir: `outputs/tal_vlm_review_epoch034_non_gt/SPG5Q_c150_t9.20_33.42/putting_drink_container_9.88_15.24`
- Evidence: No drink container is visible being placed down by the person. An orange bottle is visible on a shelf in the background but not being interacted with.

### `YVH4J_c051_t0.00_38.30` / sitting_on_chair

- Window: 12.51s - 15.90s
- Review dir: `outputs/tal_vlm_review_epoch034_non_gt/YVH4J_c051_t0.00_38.30/sitting_on_chair_12.51_15.90`
- Evidence: Person is lying on stomach on a bed, not seated on a chair.

### `QU2WL_c019_t0.00_34.21` / standing_up

- Window: 2.61s - 29.52s
- Review dir: `outputs/tal_vlm_review_epoch034_non_gt/QU2WL_c019_t0.00_34.21/standing_up_2.61_29.52`
- Evidence: The person is standing in frames 0-4 and then sits down in frame 5, which is a sitting action, not standing up.

### `1Y09V_c109_t0.00_30.79` / sitting_down

- Window: 28.45s - 31.79s
- Review dir: `outputs/tal_vlm_review_epoch034_non_gt/1Y09V_c109_t0.00_30.79/sitting_down_28.45_31.79`

### `7WIKW_c151_t17.10_42.10` / pouring_drink_container

- Window: 5.90s - 11.53s
- Review dir: `outputs/tal_vlm_review_epoch034_non_gt/7WIKW_c151_t17.10_42.10/pouring_drink_container_5.90_11.53`

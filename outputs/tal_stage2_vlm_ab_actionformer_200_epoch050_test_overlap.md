# TAL / Stage2 / VLM A/B Report

## Scope

- TAL clips: 34
- Evaluated overlap clips: 12
- GT action instances in overlap clips: 33
- Note: current TAL export covers the ActionFormer 200-clip test split (34 clips); Stage2/VLM comparison only covers clips already present in the 50-clip batch outputs.

## GT Recall Proxy

| System | GT hits | Recall proxy |
|---|---:|---:|
| TAL top-k | 18 | 54.55% |
| Stage2/VLM/Fusion | 9 | 27.27% |
| Both | 6 | - |

## Outcome Counts

| Outcome | Count |
|---|---:|
| gt_hit_by_both | 6 |
| gt_hit_by_stage2_vlm_only | 3 |
| gt_hit_by_tal_only | 12 |
| gt_missed_by_both | 12 |
| non_gt_predicted_by_both | 1 |
| not_present | 125 |
| stage2_vlm_only_non_gt | 2 |
| tal_only_non_gt | 73 |

## Per Action

| Action | GT | TAL hits | Stage2/VLM hits | TAL preds | Final | Pending | Rejected |
|---|---:|---:|---:|---:|---:|---:|---:|
| cooking | 0 | 0 | 0 | 4 | 0 | 0 | 0 |
| drinking_water | 2 | 2 | 2 | 9 | 2 | 0 | 4 |
| eating | 4 | 2 | 1 | 2 | 0 | 0 | 0 |
| holding_drink_container | 1 | 1 | 0 | 11 | 0 | 0 | 0 |
| holding_phone | 3 | 2 | 2 | 7 | 0 | 0 | 0 |
| looking_at_phone | 1 | 0 | 0 | 4 | 0 | 0 | 0 |
| pouring_drink_container | 0 | 0 | 0 | 2 | 0 | 0 | 0 |
| putting_drink_container | 1 | 0 | 0 | 1 | 0 | 0 | 0 |
| running | 2 | 1 | 0 | 2 | 0 | 0 | 0 |
| sitting_down | 3 | 2 | 0 | 6 | 0 | 0 | 0 |
| sitting_on_chair | 2 | 2 | 0 | 8 | 0 | 0 | 0 |
| sitting_on_floor | 1 | 0 | 0 | 3 | 0 | 0 | 0 |
| sitting_on_sofa | 2 | 1 | 0 | 4 | 0 | 0 | 0 |
| standing_up | 3 | 1 | 0 | 6 | 0 | 0 | 0 |
| taking_drink_container | 2 | 0 | 1 | 3 | 0 | 0 | 0 |
| taking_phone | 0 | 0 | 0 | 4 | 0 | 0 | 0 |
| talking_on_phone | 1 | 0 | 1 | 2 | 0 | 0 | 2 |
| using_laptop | 0 | 0 | 0 | 3 | 0 | 0 | 0 |
| walking_through_doorway | 3 | 3 | 1 | 7 | 0 | 0 | 0 |
| watching_laptop | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| watching_tv | 2 | 1 | 1 | 3 | 0 | 0 | 0 |

## Difference Examples

### TAL hits GT, Stage2/VLM misses

| Clip | Action | TAL score | TAL IoU | Fused status | Final/Pending/Rejected |
|---|---|---:|---:|---|---|
| `1BUFQ_c132_t0.00_40.25` | sitting_down | 0.2214 | 0.477 | missed_label | 0/0/0 |
| `3HUXR_c132_t0.00_41.62` | watching_tv | 0.1485 | 0.978 | missed_label | 0/0/0 |
| `5B8M5_c156_t0.00_30.71` | sitting_on_chair | 0.2660 | 0.926 | missed_label | 0/0/0 |
| `8BG1T_c106_t5.60_45.46` | sitting_down | 0.3211 | 0.729 | missed_label | 0/0/0 |
| `8BG1T_c106_t5.60_45.46` | sitting_on_sofa | 0.0964 | 0.753 | missed_label | 0/0/0 |
| `8BG1T_c106_t5.60_45.46` | standing_up | 0.1568 | 0.440 | missed_label | 0/0/0 |
| `K8AUX_c110_t0.00_29.54` | holding_drink_container | 0.2435 | 0.889 | missed_label | 0/0/0 |
| `K8AUX_c110_t0.00_29.54` | walking_through_doorway | 0.1127 | 0.666 | missed_label | 0/0/0 |

### Stage2/VLM hits GT, TAL misses

| Clip | Action | TAL score | TAL IoU | Fused status | Final/Pending/Rejected |
|---|---|---:|---:|---|---|
| `1BUFQ_c132_t0.00_40.25` | watching_tv | 0.0000 | 0.000 | vlm_recovered_label | 0/0/0 |
| `K8AUX_c110_t0.00_29.54` | taking_drink_container | 0.0000 | 0.000 | vlm_recovered_label | 0/0/0 |
| `OINMN_c019_t0.00_45.25` | talking_on_phone | 0.0000 | 0.000 | vlm_recovered_label | 0/0/0 |

### Both hit GT

| Clip | Action | TAL score | TAL IoU | Fused status | Final/Pending/Rejected |
|---|---|---:|---:|---|---|
| `024PD_c015_t0.00_59.04` | holding_phone | 0.1166 | 0.867 | vlm_recovered_label | 0/0/0 |
| `406LH_c097_t1.00_21.00` | walking_through_doorway | 0.1549 | 0.329 | vlm_recovered_label | 0/0/0 |
| `5B8M5_c156_t0.00_30.71` | eating | 0.0920 | 0.423 | vlm_recovered_label | 0/0/0 |
| `8BG1T_c106_t5.60_45.46` | drinking_water | 0.1550 | 0.939 | needs_temporal_review | 1/0/0 |
| `K8AUX_c110_t0.00_29.54` | drinking_water | 0.3362 | 0.862 | needs_temporal_review | 1/0/0 |
| `Y6MUU_c015_t0.00_39.00` | holding_phone | 0.0926 | 0.822 | vlm_recovered_label | 0/0/0 |


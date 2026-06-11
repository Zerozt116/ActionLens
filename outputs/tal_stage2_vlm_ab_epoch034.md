# TAL / Stage2 / VLM A/B Report

## Scope

- TAL clips: 7
- Evaluated overlap clips: 7
- GT action instances in overlap clips: 27
- Note: current TAL eval covers the ActionFormer val split only, not all 50 batch clips.

## GT Recall Proxy

| System | GT hits | Recall proxy |
|---|---:|---:|
| TAL top-k | 5 | 18.52% |
| Stage2/VLM/Fusion | 6 | 22.22% |
| Both | 0 | - |

## Outcome Counts

| Outcome | Count |
|---|---:|
| gt_hit_by_stage2_vlm_only | 6 |
| gt_hit_by_tal_only | 5 |
| gt_missed_by_both | 16 |
| non_gt_predicted_by_both | 1 |
| not_present | 89 |
| tal_only_non_gt | 18 |

## Per Action

| Action | GT | TAL hits | Stage2/VLM hits | TAL preds | Final | Pending | Rejected |
|---|---:|---:|---:|---:|---:|---:|---:|
| cooking | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| drinking_water | 3 | 0 | 3 | 1 | 6 | 0 | 16 |
| eating | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| holding_drink_container | 3 | 2 | 0 | 2 | 0 | 0 | 0 |
| holding_phone | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| pouring_drink_container | 0 | 0 | 0 | 2 | 0 | 0 | 0 |
| putting_drink_container | 3 | 0 | 0 | 3 | 0 | 0 | 0 |
| running | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| sitting_down | 3 | 0 | 1 | 2 | 0 | 0 | 0 |
| sitting_on_chair | 1 | 1 | 0 | 4 | 0 | 0 | 0 |
| sitting_on_floor | 3 | 0 | 0 | 1 | 0 | 0 | 0 |
| standing_up | 1 | 1 | 0 | 5 | 0 | 0 | 0 |
| taking_drink_container | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| talking_on_phone | 1 | 0 | 1 | 0 | 5 | 0 | 0 |
| walking_through_doorway | 1 | 1 | 0 | 3 | 0 | 0 | 0 |
| watching_laptop | 1 | 0 | 1 | 0 | 0 | 0 | 0 |
| watching_tv | 1 | 0 | 0 | 0 | 0 | 0 | 0 |

## Difference Examples

### TAL hits GT, Stage2/VLM misses

| Clip | Action | TAL score | TAL IoU | Fused status | Final/Pending/Rejected |
|---|---|---:|---:|---|---|
| `1Y09V_c109_t0.00_30.79` | holding_drink_container | 0.0444 | 0.671 | missed_label | 0/0/0 |
| `38TF8_c106_t0.00_31.50` | holding_drink_container | 0.0436 | 0.341 | missed_label | 0/0/0 |
| `7WIKW_c151_t17.10_42.10` | standing_up | 0.0340 | 0.000 | missed_label | 0/0/0 |
| `7WIKW_c151_t17.10_42.10` | walking_through_doorway | 0.0347 | 0.038 | missed_label | 0/0/0 |
| `QU2WL_c019_t0.00_34.21` | sitting_on_chair | 0.0502 | 0.000 | missed_label | 0/0/0 |

### Stage2/VLM hits GT, TAL misses

| Clip | Action | TAL score | TAL IoU | Fused status | Final/Pending/Rejected |
|---|---|---:|---:|---|---|
| `1Y09V_c109_t0.00_30.79` | drinking_water | 0.0000 | 0.000 | confirmed_event | 5/0/0 |
| `38TF8_c106_t0.00_31.50` | drinking_water | 0.0000 | 0.000 | needs_temporal_review | 1/0/8 |
| `7WIKW_c151_t17.10_42.10` | sitting_down | 0.0000 | 0.000 | vlm_recovered_label | 0/0/0 |
| `QU2WL_c019_t0.00_34.21` | drinking_water | 0.0000 | 0.000 | needs_temporal_review | 0/0/7 |
| `QU2WL_c019_t0.00_34.21` | talking_on_phone | 0.0000 | 0.000 | confirmed_event | 5/0/0 |
| `YVH4J_c051_t0.00_38.30` | watching_laptop | 0.0000 | 0.000 | vlm_recovered_label | 0/0/0 |

### Both hit GT

- None


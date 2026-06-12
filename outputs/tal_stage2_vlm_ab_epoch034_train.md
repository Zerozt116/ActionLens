# TAL / Stage2 / VLM A/B Report

## Scope

- TAL clips: 32
- Evaluated overlap clips: 32
- GT action instances in overlap clips: 84
- Note: current TAL export covers the ActionFormer train split (32 clips); metrics are optimistic because this split was used for training.

## GT Recall Proxy

| System | GT hits | Recall proxy |
|---|---:|---:|
| TAL top-k | 38 | 45.24% |
| Stage2/VLM/Fusion | 16 | 19.05% |
| Both | 4 | - |

## Outcome Counts

| Outcome | Count |
|---|---:|
| gt_hit_by_both | 4 |
| gt_hit_by_stage2_vlm_only | 12 |
| gt_hit_by_tal_only | 34 |
| gt_missed_by_both | 34 |
| not_present | 451 |
| stage2_vlm_only_non_gt | 13 |
| tal_only_non_gt | 70 |

## Per Action

| Action | GT | TAL hits | Stage2/VLM hits | TAL preds | Final | Pending | Rejected |
|---|---:|---:|---:|---:|---:|---:|---:|
| cooking | 3 | 2 | 0 | 7 | 0 | 0 | 0 |
| drinking_water | 6 | 2 | 2 | 4 | 3 | 0 | 27 |
| eating | 8 | 3 | 3 | 3 | 0 | 0 | 0 |
| holding_drink_container | 8 | 1 | 1 | 7 | 0 | 0 | 0 |
| holding_phone | 6 | 2 | 2 | 3 | 0 | 0 | 0 |
| looking_at_phone | 5 | 0 | 2 | 0 | 0 | 0 | 0 |
| pouring_drink_container | 0 | 0 | 0 | 10 | 0 | 0 | 0 |
| putting_drink_container | 4 | 3 | 1 | 12 | 0 | 0 | 0 |
| running | 3 | 3 | 0 | 3 | 0 | 0 | 0 |
| sitting_down | 7 | 3 | 0 | 7 | 0 | 0 | 0 |
| sitting_on_chair | 7 | 6 | 1 | 20 | 0 | 0 | 0 |
| sitting_on_floor | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| sitting_on_sofa | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| standing_up | 9 | 9 | 0 | 22 | 0 | 0 | 0 |
| taking_drink_container | 5 | 0 | 1 | 0 | 0 | 0 | 0 |
| talking_on_phone | 1 | 0 | 1 | 0 | 0 | 0 | 26 |
| using_laptop | 1 | 0 | 1 | 0 | 0 | 0 | 0 |
| walking_through_doorway | 4 | 4 | 0 | 10 | 0 | 0 | 0 |
| watching_laptop | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| watching_tv | 1 | 0 | 1 | 0 | 0 | 0 | 0 |

## Difference Examples

### TAL hits GT, Stage2/VLM misses

| Clip | Action | TAL score | TAL IoU | Fused status | Final/Pending/Rejected |
|---|---|---:|---:|---|---|
| `1BUFQ_c132_t0.00_40.25` | standing_up | 0.0472 | 0.449 | missed_label | 0/0/0 |
| `24B2K_c125_t6.50_51.50` | putting_drink_container | 0.0457 | 0.084 | missed_label | 0/0/0 |
| `24B2K_c125_t6.50_51.50` | standing_up | 0.0394 | 0.000 | missed_label | 0/0/0 |
| `5B8M5_c156_t0.00_30.71` | sitting_down | 0.0396 | 0.541 | missed_label | 0/0/0 |
| `5B8M5_c156_t0.00_30.71` | sitting_on_chair | 0.0441 | 0.787 | missed_label | 0/0/0 |
| `5B8M5_c156_t0.00_30.71` | standing_up | 0.0552 | 0.470 | missed_label | 0/0/0 |
| `75HWR_c016_t0.00_37.00` | holding_phone | 0.0307 | 0.025 | missed_label | 0/0/0 |
| `75HWR_c016_t0.00_37.00` | running | 0.0396 | 0.287 | missed_label | 0/0/0 |

### Stage2/VLM hits GT, TAL misses

| Clip | Action | TAL score | TAL IoU | Fused status | Final/Pending/Rejected |
|---|---|---:|---:|---|---|
| `1BUFQ_c132_t0.00_40.25` | watching_tv | 0.0000 | 0.000 | vlm_recovered_label | 0/0/0 |
| `5B8M5_c156_t0.00_30.71` | eating | 0.0000 | 0.000 | vlm_recovered_label | 0/0/0 |
| `75HWR_c016_t0.00_37.00` | looking_at_phone | 0.0000 | 0.000 | vlm_recovered_label | 0/0/0 |
| `7JTEK_c156_t0.00_35.29` | eating | 0.0000 | 0.000 | vlm_recovered_label | 0/0/0 |
| `EHTB6_c016_t0.00_41.00` | looking_at_phone | 0.0000 | 0.000 | vlm_recovered_label | 0/0/0 |
| `ENJ7V_c015_t0.00_39.79` | holding_phone | 0.0000 | 0.000 | vlm_recovered_label | 0/0/0 |
| `K8AUX_c110_t0.00_29.54` | drinking_water | 0.0000 | 0.000 | needs_temporal_review | 1/0/0 |
| `K8AUX_c110_t0.00_29.54` | taking_drink_container | 0.0000 | 0.000 | vlm_recovered_label | 0/0/0 |

### Both hit GT

| Clip | Action | TAL score | TAL IoU | Fused status | Final/Pending/Rejected |
|---|---|---:|---:|---|---|
| `229ZR_c059_t0.00_40.42` | sitting_on_chair | 0.0561 | 0.630 | vlm_recovered_label | 0/0/0 |
| `GY9MZ_c107_t0.00_35.17` | holding_drink_container | 0.0387 | 0.690 | vlm_recovered_label | 0/0/0 |
| `R28EY_c065_t0.00_30.67` | eating | 0.0399 | 0.685 | vlm_recovered_label | 0/0/0 |
| `U9UI8_c106_t0.00_30.54` | drinking_water | 0.0317 | 0.361 | confirmed_event | 2/0/0 |


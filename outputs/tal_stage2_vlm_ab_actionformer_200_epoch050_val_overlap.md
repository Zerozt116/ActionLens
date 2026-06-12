# TAL / Stage2 / VLM A/B Report

## Scope

- TAL clips: 29
- Evaluated overlap clips: 5
- GT action instances in overlap clips: 12
- Note: current TAL export covers the ActionFormer 200-clip val split (29 clips); Stage2/VLM comparison only covers clips already present in the 50-clip batch outputs.

## GT Recall Proxy

| System | GT hits | Recall proxy |
|---|---:|---:|
| TAL top-k | 7 | 58.33% |
| Stage2/VLM/Fusion | 3 | 25.00% |
| Both | 2 | - |

## Outcome Counts

| Outcome | Count |
|---|---:|
| gt_hit_by_both | 2 |
| gt_hit_by_stage2_vlm_only | 1 |
| gt_hit_by_tal_only | 5 |
| gt_missed_by_both | 4 |
| non_gt_predicted_by_both | 2 |
| not_present | 57 |
| stage2_vlm_only_non_gt | 1 |
| tal_only_non_gt | 25 |

## Per Action

| Action | GT | TAL hits | Stage2/VLM hits | TAL preds | Final | Pending | Rejected |
|---|---:|---:|---:|---:|---:|---:|---:|
| cooking | 2 | 2 | 1 | 2 | 0 | 0 | 0 |
| drinking_water | 0 | 0 | 0 | 4 | 0 | 0 | 7 |
| eating | 1 | 0 | 1 | 1 | 0 | 0 | 0 |
| holding_drink_container | 1 | 1 | 0 | 3 | 0 | 0 | 0 |
| holding_phone | 2 | 1 | 1 | 3 | 0 | 0 | 0 |
| looking_at_phone | 2 | 1 | 0 | 1 | 0 | 0 | 0 |
| pouring_drink_container | 0 | 0 | 0 | 2 | 0 | 0 | 0 |
| putting_drink_container | 1 | 0 | 0 | 1 | 0 | 0 | 0 |
| sitting_down | 1 | 0 | 0 | 1 | 0 | 0 | 0 |
| sitting_on_chair | 0 | 0 | 0 | 3 | 0 | 0 | 0 |
| standing_up | 2 | 2 | 0 | 5 | 0 | 0 | 0 |
| taking_drink_container | 0 | 0 | 0 | 2 | 0 | 0 | 0 |
| talking_on_phone | 0 | 0 | 0 | 1 | 0 | 0 | 12 |
| using_laptop | 0 | 0 | 0 | 2 | 0 | 0 | 0 |
| walking_through_doorway | 0 | 0 | 0 | 2 | 0 | 0 | 0 |
| watching_laptop | 0 | 0 | 0 | 1 | 0 | 0 | 0 |

## Difference Examples

### TAL hits GT, Stage2/VLM misses

| Clip | Action | TAL score | TAL IoU | Fused status | Final/Pending/Rejected |
|---|---|---:|---:|---|---|
| `7JTEK_c156_t0.00_35.29` | cooking | 0.2996 | 0.862 | missed_label | 0/0/0 |
| `7JTEK_c156_t0.00_35.29` | holding_drink_container | 0.1849 | 0.293 | missed_label | 0/0/0 |
| `ENJ7V_c015_t0.00_39.79` | looking_at_phone | 0.0774 | 0.856 | missed_label | 0/0/0 |
| `FV684_c154_t0.00_32.92` | standing_up | 0.1741 | 0.976 | missed_label | 0/0/0 |
| `P0DXX_c151_t6.20_33.40` | standing_up | 0.2208 | 0.194 | missed_label | 0/0/0 |

### Stage2/VLM hits GT, TAL misses

| Clip | Action | TAL score | TAL IoU | Fused status | Final/Pending/Rejected |
|---|---|---:|---:|---|---|
| `7JTEK_c156_t0.00_35.29` | eating | 0.0000 | 0.000 | vlm_recovered_label | 0/0/0 |

### Both hit GT

| Clip | Action | TAL score | TAL IoU | Fused status | Final/Pending/Rejected |
|---|---|---:|---:|---|---|
| `BPZE3_c147_t0.00_35.12` | cooking | 0.1172 | 0.844 | vlm_recovered_label | 0/0/0 |
| `ENJ7V_c015_t0.00_39.79` | holding_phone | 0.1867 | 0.856 | vlm_recovered_label | 0/0/0 |


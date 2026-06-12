# TAL / Stage2 / VLM A/B Report

## Scope

- TAL clips: 11
- Evaluated overlap clips: 11
- GT action instances in overlap clips: 28
- Note: current TAL export covers the ActionFormer test split (11 clips); this split was not used for training.

## GT Recall Proxy

| System | GT hits | Recall proxy |
|---|---:|---:|
| TAL top-k | 5 | 17.86% |
| Stage2/VLM/Fusion | 8 | 28.57% |
| Both | 2 | - |

## Outcome Counts

| Outcome | Count |
|---|---:|
| gt_hit_by_both | 2 |
| gt_hit_by_stage2_vlm_only | 6 |
| gt_hit_by_tal_only | 3 |
| gt_missed_by_both | 17 |
| not_present | 152 |
| stage2_vlm_only_non_gt | 3 |
| tal_only_non_gt | 29 |

## Per Action

| Action | GT | TAL hits | Stage2/VLM hits | TAL preds | Final | Pending | Rejected |
|---|---:|---:|---:|---:|---:|---:|---:|
| cooking | 1 | 0 | 1 | 0 | 0 | 0 | 0 |
| drinking_water | 2 | 0 | 2 | 0 | 6 | 0 | 1 |
| eating | 3 | 0 | 0 | 1 | 0 | 0 | 0 |
| holding_drink_container | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| holding_phone | 4 | 0 | 1 | 0 | 0 | 0 | 0 |
| looking_at_phone | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| pouring_drink_container | 0 | 0 | 0 | 3 | 0 | 0 | 0 |
| putting_drink_container | 0 | 0 | 0 | 6 | 0 | 0 | 0 |
| running | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| sitting_down | 2 | 0 | 0 | 4 | 0 | 0 | 0 |
| sitting_on_chair | 2 | 2 | 1 | 9 | 0 | 0 | 0 |
| sitting_on_floor | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| sitting_on_sofa | 2 | 0 | 1 | 0 | 0 | 0 | 0 |
| standing_up | 4 | 2 | 0 | 7 | 0 | 0 | 0 |
| talking_on_phone | 1 | 0 | 1 | 0 | 0 | 0 | 1 |
| using_laptop | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| walking_through_doorway | 1 | 1 | 1 | 2 | 0 | 0 | 0 |
| watching_laptop | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| watching_tv | 1 | 0 | 0 | 0 | 0 | 0 | 0 |

## Difference Examples

### TAL hits GT, Stage2/VLM misses

| Clip | Action | TAL score | TAL IoU | Fused status | Final/Pending/Rejected |
|---|---|---:|---:|---|---|
| `4FXUI_c052_t0.00_43.20` | sitting_on_chair | 0.0395 | 0.083 | missed_label | 0/0/0 |
| `8BG1T_c106_t5.60_45.46` | standing_up | 0.0363 | 0.000 | missed_label | 0/0/0 |
| `ARCUY_c059_t0.60_43.83` | standing_up | 0.0420 | 0.250 | missed_label | 0/0/0 |

### Stage2/VLM hits GT, TAL misses

| Clip | Action | TAL score | TAL IoU | Fused status | Final/Pending/Rejected |
|---|---|---:|---:|---|---|
| `024PD_c015_t0.00_59.04` | holding_phone | 0.0000 | 0.000 | vlm_recovered_label | 0/0/0 |
| `8BG1T_c106_t5.60_45.46` | drinking_water | 0.0000 | 0.000 | needs_temporal_review | 1/0/0 |
| `BPZE3_c147_t0.00_35.12` | cooking | 0.0000 | 0.000 | vlm_recovered_label | 0/0/0 |
| `OINMN_c019_t0.00_45.25` | talking_on_phone | 0.0000 | 0.000 | vlm_recovered_label | 0/0/0 |
| `VSFCR_c123_t0.00_37.88` | drinking_water | 0.0000 | 0.000 | needs_temporal_review | 5/0/0 |
| `VSFCR_c123_t0.00_37.88` | sitting_on_sofa | 0.0000 | 0.000 | vlm_recovered_label | 0/0/0 |

### Both hit GT

| Clip | Action | TAL score | TAL IoU | Fused status | Final/Pending/Rejected |
|---|---|---:|---:|---|---|
| `406LH_c097_t1.00_21.00` | walking_through_doorway | 0.0279 | 0.616 | vlm_recovered_label | 0/0/0 |
| `ARCUY_c059_t0.60_43.83` | sitting_on_chair | 0.0351 | 0.032 | vlm_recovered_label | 0/0/0 |


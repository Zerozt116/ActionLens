# TAL Proposal VLM Review

## Summary

| Metric | Value |
|---|---:|
| proposals_selected | 5 |
| completed | 5 |
| failed | 0 |
| vlm_confirmed | 2 |
| vlm_weak_confirmed | 0 |
| vlm_rejected | 3 |
| dry_run | 0 |

## Reviews

| Clip | Action | Outcome | TAL score | TAL IoU | VLM decision | Confidence |
|---|---|---|---:|---:|---|---:|
| `1Y09V_c109_t0.00_30.79` | holding_drink_container | gt_hit_by_tal_only | 0.0444 | 0.671 | vlm_confirmed | 0.95 |
| `38TF8_c106_t0.00_31.50` | holding_drink_container | gt_hit_by_tal_only | 0.0436 | 0.341 | vlm_confirmed | 0.95 |
| `7WIKW_c151_t17.10_42.10` | walking_through_doorway | gt_hit_by_tal_only | 0.0347 | 0.038 | vlm_rejected | 0.00 |
| `QU2WL_c019_t0.00_34.21` | sitting_on_chair | gt_hit_by_tal_only | 0.0502 | 0.000 | vlm_rejected | 0.00 |
| `7WIKW_c151_t17.10_42.10` | standing_up | gt_hit_by_tal_only | 0.0340 | 0.000 | vlm_rejected | 0.00 |

## Evidence

### `1Y09V_c109_t0.00_30.79` / holding_drink_container

- Window: 0.00s - 19.84s
- Review dir: `outputs/tal_vlm_review_epoch034/1Y09V_c109_t0.00_30.79/holding_drink_container_0.00_19.84`
- Evidence: The person is holding a white cup on a saucer in multiple frames, including frame 0, 1, 2, 3, 4, and 5.

### `38TF8_c106_t0.00_31.50` / holding_drink_container

- Window: 0.00s - 4.37s
- Review dir: `outputs/tal_vlm_review_epoch034/38TF8_c106_t0.00_31.50/holding_drink_container_0.00_4.37`
- Evidence: The man in the background is holding a white cup in his hand across multiple frames (0-5).

### `7WIKW_c151_t17.10_42.10` / walking_through_doorway

- Window: 1.23s - 25.46s
- Review dir: `outputs/tal_vlm_review_epoch034/7WIKW_c151_t17.10_42.10/walking_through_doorway_1.23_25.46`
- Evidence: Person is sitting on a box near the doorway, not walking through it.

### `QU2WL_c019_t0.00_34.21` / sitting_on_chair

- Window: 3.73s - 7.20s
- Review dir: `outputs/tal_vlm_review_epoch034/QU2WL_c019_t0.00_34.21/sitting_on_chair_3.73_7.20`
- Evidence: The person is standing throughout the sampled frames, holding a phone to their ear.

### `7WIKW_c151_t17.10_42.10` / standing_up

- Window: 1.17s - 4.90s
- Review dir: `outputs/tal_vlm_review_epoch034/7WIKW_c151_t17.10_42.10/standing_up_1.17_4.90`
- Evidence: Person is already standing on a box, holding a pillow, no transition from sitting or crouching visible.

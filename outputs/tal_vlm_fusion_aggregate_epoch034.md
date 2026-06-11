# TAL+VLM Fusion Aggregate

## Summary

| Metric | Value |
|---|---:|
| evaluated_clip_count | 7 |
| gt_action_instances | 27 |
| baseline_stage2_vlm_gt_hits | 6 |
| tal_vlm_confirmed_events | 2 |
| tal_vlm_confirmed_gt_hits | 2 |
| tal_vlm_added_gt_hits | 2 |
| tal_vlm_rejected_events | 3 |
| combined_gt_hits | 8 |
| baseline_recall_proxy | 0.2222 |
| combined_recall_proxy | 0.2963 |
| absolute_recall_gain | 0.0741 |
| relative_hit_gain | 0.3333 |

## Per Action

| Action | GT | Baseline hits | TAL+VLM GT hits | Added GT hits | Confirmed | Rejected |
|---|---:|---:|---:|---:|---:|---:|
| drinking_water | 3 | 3 | 0 | 0 | 0 | 0 |
| eating | 2 | 0 | 0 | 0 | 0 | 0 |
| holding_drink_container | 3 | 0 | 2 | 2 | 2 | 0 |
| holding_phone | 1 | 0 | 0 | 0 | 0 | 0 |
| putting_drink_container | 3 | 0 | 0 | 0 | 0 | 0 |
| running | 1 | 0 | 0 | 0 | 0 | 0 |
| sitting_down | 3 | 1 | 0 | 0 | 0 | 0 |
| sitting_on_chair | 1 | 0 | 0 | 0 | 0 | 1 |
| sitting_on_floor | 3 | 0 | 0 | 0 | 0 | 0 |
| standing_up | 1 | 0 | 0 | 0 | 0 | 1 |
| taking_drink_container | 2 | 0 | 0 | 0 | 0 | 0 |
| talking_on_phone | 1 | 1 | 0 | 0 | 0 | 0 |
| walking_through_doorway | 1 | 0 | 0 | 0 | 0 | 1 |
| watching_laptop | 1 | 1 | 0 | 0 | 0 | 0 |
| watching_tv | 1 | 0 | 0 | 0 | 0 | 0 |

## Added GT Hits

| Clip | Action | Time | TAL score | VLM confidence | Evidence |
|---|---|---|---:|---:|---|
| `1Y09V_c109_t0.00_30.79` | holding_drink_container | 0.00-19.84s | 0.0444 | 0.95 | The person is holding a white cup on a saucer in multiple frames, including frame 0, 1, 2, 3, 4, and 5. |
| `38TF8_c106_t0.00_31.50` | holding_drink_container | 0.00-4.37s | 0.0436 | 0.95 | The man in the background is holding a white cup in his hand across multiple frames (0-5). |

## Rejected TAL Proposals

| Clip | Action | GT? | TAL score | Evidence |
|---|---|---|---:|---|
| `7WIKW_c151_t17.10_42.10` | walking_through_doorway | yes | 0.0347 | Person is sitting on a box near the doorway, not walking through it. |
| `QU2WL_c019_t0.00_34.21` | sitting_on_chair | yes | 0.0502 | The person is standing throughout the sampled frames, holding a phone to their ear. |
| `7WIKW_c151_t17.10_42.10` | standing_up | yes | 0.0340 | Person is already standing on a box, holding a pillow, no transition from sitting or crouching visible. |

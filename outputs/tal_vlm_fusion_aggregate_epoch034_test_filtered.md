# TAL+VLM Fusion Aggregate

## Summary

| Metric | Value |
|---|---:|
| evaluated_clip_count | 11 |
| gt_action_instances | 28 |
| baseline_stage2_vlm_gt_hits | 8 |
| tal_vlm_confirmed_events | 0 |
| tal_vlm_confirmed_gt_hits | 0 |
| tal_vlm_added_gt_hits | 0 |
| tal_vlm_rejected_events | 5 |
| combined_gt_hits | 8 |
| baseline_recall_proxy | 0.2857 |
| combined_recall_proxy | 0.2857 |
| absolute_recall_gain | 0.0000 |
| relative_hit_gain | 0.0000 |

## Per Action

| Action | GT | Baseline hits | TAL+VLM GT hits | Added GT hits | Confirmed | Rejected |
|---|---:|---:|---:|---:|---:|---:|
| cooking | 1 | 1 | 0 | 0 | 0 | 0 |
| drinking_water | 2 | 2 | 0 | 0 | 0 | 0 |
| eating | 3 | 0 | 0 | 0 | 0 | 0 |
| holding_drink_container | 1 | 0 | 0 | 0 | 0 | 0 |
| holding_phone | 4 | 1 | 0 | 0 | 0 | 0 |
| looking_at_phone | 2 | 0 | 0 | 0 | 0 | 0 |
| pouring_drink_container | 0 | 0 | 0 | 0 | 0 | 1 |
| putting_drink_container | 0 | 0 | 0 | 0 | 0 | 1 |
| sitting_down | 2 | 0 | 0 | 0 | 0 | 0 |
| sitting_on_chair | 2 | 1 | 0 | 0 | 0 | 1 |
| sitting_on_sofa | 2 | 1 | 0 | 0 | 0 | 0 |
| standing_up | 4 | 0 | 0 | 0 | 0 | 2 |
| talking_on_phone | 1 | 1 | 0 | 0 | 0 | 0 |
| using_laptop | 1 | 0 | 0 | 0 | 0 | 0 |
| walking_through_doorway | 1 | 1 | 0 | 0 | 0 | 0 |
| watching_laptop | 1 | 0 | 0 | 0 | 0 | 0 |
| watching_tv | 1 | 0 | 0 | 0 | 0 | 0 |

## Added GT Hits

- None

## Rejected TAL Proposals

| Clip | Action | GT? | TAL score | Evidence |
|---|---|---|---:|---|
| `ARCUY_c059_t0.60_43.83` | standing_up | yes | 0.0420 | The person remains seated in a chair throughout the sampled frames, interacting with a large dark object. No transition to standing is visible. |
| `VSFCR_c123_t0.00_37.88` | standing_up | no | 0.0513 | Person remains seated on the couch throughout the frames, holding a purple cup. |
| `1LKPL_c154_t0.00_33.58` | sitting_on_chair | no | 0.0504 | The person is standing while drinking from a cup. |
| `024PD_c015_t0.00_59.04` | putting_drink_container | no | 0.0415 | Person is holding an object in hands, but it is not clearly identifiable as a drink container, and no clear action of placing it down is visible. |
| `024PD_c015_t0.00_59.04` | pouring_drink_container | no | 0.0413 | Person is holding a tablet, not a drink container, and no pouring action is visible. |

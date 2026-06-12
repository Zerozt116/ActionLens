# TAL+VLM Fusion Aggregate

## Summary

| Metric | Value |
|---|---:|
| evaluated_clip_count | 12 |
| gt_action_instances | 33 |
| baseline_stage2_vlm_gt_hits | 9 |
| tal_vlm_confirmed_events | 4 |
| tal_vlm_confirmed_gt_hits | 3 |
| tal_vlm_added_gt_hits | 3 |
| tal_vlm_rejected_events | 12 |
| combined_gt_hits | 12 |
| baseline_recall_proxy | 0.2727 |
| combined_recall_proxy | 0.3636 |
| absolute_recall_gain | 0.0909 |
| relative_hit_gain | 0.3333 |

## Per Action

| Action | GT | Baseline hits | TAL+VLM GT hits | Added GT hits | Confirmed | Rejected |
|---|---:|---:|---:|---:|---:|---:|
| drinking_water | 2 | 2 | 0 | 0 | 0 | 1 |
| eating | 4 | 1 | 0 | 0 | 0 | 0 |
| holding_drink_container | 1 | 0 | 1 | 1 | 2 | 2 |
| holding_phone | 3 | 2 | 0 | 0 | 0 | 2 |
| looking_at_phone | 1 | 0 | 0 | 0 | 0 | 1 |
| pouring_drink_container | 0 | 0 | 0 | 0 | 0 | 1 |
| putting_drink_container | 1 | 0 | 0 | 0 | 0 | 0 |
| running | 2 | 0 | 0 | 0 | 0 | 0 |
| sitting_down | 3 | 0 | 1 | 1 | 1 | 1 |
| sitting_on_chair | 2 | 0 | 1 | 1 | 1 | 1 |
| sitting_on_floor | 1 | 0 | 0 | 0 | 0 | 0 |
| sitting_on_sofa | 2 | 0 | 0 | 0 | 0 | 0 |
| standing_up | 3 | 0 | 0 | 0 | 0 | 0 |
| taking_drink_container | 2 | 1 | 0 | 0 | 0 | 0 |
| taking_phone | 0 | 0 | 0 | 0 | 0 | 1 |
| talking_on_phone | 1 | 1 | 0 | 0 | 0 | 0 |
| walking_through_doorway | 3 | 1 | 0 | 0 | 0 | 2 |
| watching_tv | 2 | 1 | 0 | 0 | 0 | 0 |

## Added GT Hits

| Clip | Action | Time | TAL score | VLM confidence | Evidence |
|---|---|---|---:|---:|---|
| `5B8M5_c156_t0.00_30.71` | sitting_on_chair | 0.00-29.56s | 0.2660 | 0.95 | The person is visibly seated on a blue chair in frames 0, 1, 2, 4, and 5. In frame 3, the person is partially occluded but still appears to be seated. |
| `8BG1T_c106_t5.60_45.46` | sitting_down | 2.30-14.02s | 0.3211 | 0.95 | The person is seen standing at frame 0, then bending and lowering themselves onto the bench by frame 3, and is fully seated by frame 4. |
| `K8AUX_c110_t0.00_29.54` | holding_drink_container | 0.00-27.36s | 0.2435 | 0.95 | The person is holding a blue cup in frames 0-2, actively drinking from it. |

## Rejected TAL Proposals

| Clip | Action | GT? | TAL score | Evidence |
|---|---|---|---:|---|
| `K8AUX_c110_t0.00_29.54` | walking_through_doorway | yes | 0.1127 | The person is standing in the doorway holding items and does not appear to be moving through it. The person is stationary or moving very slowly within the doorway area. |
| `1BUFQ_c132_t0.00_40.25` | sitting_on_chair | no | 0.3467 | The person is standing throughout the sampled frames. |
| `1BUFQ_c132_t0.00_40.25` | holding_phone | no | 0.2392 |  |
| `406LH_c097_t1.00_21.00` | sitting_down | no | 0.2195 | Person is standing throughout the sampled frames, interacting with a door. |
| `406LH_c097_t1.00_21.00` | pouring_drink_container | no | 0.1759 | No drink container or pouring action is visible. Person is holding a phone and opening a door. |
| `SPG5Q_c150_t9.20_33.42` | drinking_water | no | 0.1576 | The person is bending over and interacting with items on the counter, but no cup, bottle, or drinking action is visible. |
| `SPG5Q_c150_t9.20_33.42` | holding_drink_container | no | 0.1568 |  |
| `5B8M5_c156_t0.00_30.71` | looking_at_phone | no | 0.1458 | No phone is visible in any of the sampled frames. |
| `S1BYH_c150_t0.00_28.04` | holding_drink_container | no | 0.1319 |  |
| `5B8M5_c156_t0.00_30.71` | holding_phone | no | 0.1304 |  |
| `S1BYH_c150_t0.00_28.04` | taking_phone | no | 0.0974 |  |
| `S1BYH_c150_t0.00_28.04` | walking_through_doorway | no | 0.0970 | No doorway is visible in any of the sampled frames. The person is moving within a cluttered garage space, but there is no evidence of a doorway or crossing one. |

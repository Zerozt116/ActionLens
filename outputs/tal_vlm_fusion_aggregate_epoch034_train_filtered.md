# TAL+VLM Fusion Aggregate

## Summary

| Metric | Value |
|---|---:|
| evaluated_clip_count | 32 |
| gt_action_instances | 84 |
| baseline_stage2_vlm_gt_hits | 16 |
| tal_vlm_confirmed_events | 11 |
| tal_vlm_confirmed_gt_hits | 10 |
| tal_vlm_added_gt_hits | 10 |
| tal_vlm_rejected_events | 11 |
| combined_gt_hits | 26 |
| baseline_recall_proxy | 0.1905 |
| combined_recall_proxy | 0.3095 |
| absolute_recall_gain | 0.1190 |
| relative_hit_gain | 0.6250 |

## Per Action

| Action | GT | Baseline hits | TAL+VLM GT hits | Added GT hits | Confirmed | Rejected |
|---|---:|---:|---:|---:|---:|---:|
| cooking | 3 | 0 | 0 | 0 | 0 | 0 |
| drinking_water | 6 | 2 | 0 | 0 | 0 | 0 |
| eating | 8 | 3 | 0 | 0 | 0 | 0 |
| holding_drink_container | 8 | 1 | 0 | 0 | 1 | 0 |
| holding_phone | 6 | 2 | 0 | 0 | 0 | 1 |
| looking_at_phone | 5 | 2 | 0 | 0 | 0 | 0 |
| putting_drink_container | 4 | 1 | 0 | 0 | 0 | 2 |
| running | 3 | 0 | 0 | 0 | 0 | 0 |
| sitting_down | 7 | 0 | 1 | 1 | 1 | 0 |
| sitting_on_chair | 7 | 1 | 4 | 4 | 4 | 4 |
| sitting_on_floor | 4 | 0 | 0 | 0 | 0 | 0 |
| sitting_on_sofa | 1 | 0 | 0 | 0 | 0 | 0 |
| standing_up | 9 | 0 | 4 | 4 | 4 | 4 |
| taking_drink_container | 5 | 1 | 0 | 0 | 0 | 0 |
| talking_on_phone | 1 | 1 | 0 | 0 | 0 | 0 |
| using_laptop | 1 | 1 | 0 | 0 | 0 | 0 |
| walking_through_doorway | 4 | 0 | 1 | 1 | 1 | 0 |
| watching_laptop | 1 | 0 | 0 | 0 | 0 | 0 |
| watching_tv | 1 | 1 | 0 | 0 | 0 | 0 |

## Added GT Hits

| Clip | Action | Time | TAL score | VLM confidence | Evidence |
|---|---|---|---:|---:|---|
| `1BUFQ_c132_t0.00_40.25` | standing_up | 33.73-38.79s | 0.0472 | 0.95 | Person is seen leaning forward from a seated position, then rising to a standing position while holding a piece of paper. |
| `5B8M5_c156_t0.00_30.71` | sitting_on_chair | 0.00-25.18s | 0.0441 | 0.95 | The person is visibly seated on a blue chair in frames 0, 1, 2, and 5. The chair is partially obscured in frame 3 but the person's posture indicates sitting. In frame 4, the person is standing, so sitting is not occurring at that moment. |
| `CY2J2_c059_t0.00_50.83` | standing_up | 42.09-51.30s | 0.0401 | 0.95 | In frame 4, the person is seen rising from a seated position, and by frame 5, they are fully standing. This matches the definition of standing_up. |
| `FV684_c154_t0.00_32.92` | standing_up | 4.62-31.25s | 0.0542 | 0.95 | The person transitions from a bent-over position at the sink to standing upright while holding a vacuum cleaner. |
| `M8CDW_c019_t0.00_32.79` | sitting_on_chair | 4.90-19.83s | 0.0455 | 1.00 | Person is seated on a chair throughout all frames, with legs extended and body supported by the chair. |
| `PCXYE_c109_t0.00_28.40` | standing_up | 0.00-24.09s | 0.0498 | 0.95 | The person is seen transitioning from sitting on a motorcycle to standing and walking away. |
| `X95MU_c016_t0.00_36.38` | sitting_down | 9.17-12.71s | 0.0506 | 0.95 | The person starts bent over at frame 0, transitions to sitting on the chair by frame 2, and is fully seated by frame 3. The motion is continuous and matches the definition of sitting down. |
| `X95MU_c016_t0.00_36.38` | sitting_on_chair | 11.44-32.86s | 0.0450 | 1.00 | The man is consistently seated on a white chair across all frames, holding a camera. |
| `Y6MUU_c015_t0.00_39.00` | walking_through_doorway | 29.63-33.40s | 0.0434 | 0.90 | The man turns his back to the camera and walks toward a door on the left side of the frame, appearing to approach or enter through it. |
| `ZP5TG_c123_t0.00_42.46` | sitting_on_chair | 9.53-39.99s | 0.0496 | 1.00 | The person in the yellow shirt is seated on a wooden chair with their legs crossed and arms resting on the chair's back and armrests. |

## Rejected TAL Proposals

| Clip | Action | GT? | TAL score | Evidence |
|---|---|---|---:|---|
| `NDH24_c052_t0.00_36.38` | holding_phone | yes | 0.0566 | Person is holding a laptop, not a phone. |
| `5B8M5_c156_t0.00_30.71` | standing_up | yes | 0.0552 | The person remains seated throughout the frames, leaning forward to interact with a plate. |
| `24B2K_c125_t6.50_51.50` | putting_drink_container | yes | 0.0457 | The person is drinking from a glass, not putting it down. |
| `FVPMC_c154_t0.00_35.25` | standing_up | yes | 0.0447 | Person is bent over inside a cabinet, not transitioning to standing. |
| `7JTEK_c156_t0.00_35.29` | putting_drink_container | yes | 0.0448 | No drink container (cup, glass, bottle, can) is visible in the frames. The person is holding a small plate or bowl, not a drink container. |
| `7JTEK_c156_t0.00_35.29` | standing_up | no | 0.0454 | Person is already standing throughout the sampled frames, no transition from sitting or crouching observed. |
| `M0DAY_c150_t0.00_24.83` | sitting_on_chair | no | 0.0445 | Person is standing on the floor, reaching into a closet. |
| `6TNP4_c107_t0.90_44.58` | sitting_on_chair | no | 0.0421 | The boy is sitting on the floor, not on a chair. |
| `BGX4T_c097_t0.00_23.42` | standing_up | no | 0.0418 | Person is already standing and holding a vacuum cleaner throughout the frames. |
| `STAZI_c097_t0.00_19.10` | sitting_on_chair | no | 0.0407 |  |
| `BGX4T_c097_t0.00_23.42` | sitting_on_chair | no | 0.0402 | Person is standing and holding a yellow steam cleaner in all frames. |

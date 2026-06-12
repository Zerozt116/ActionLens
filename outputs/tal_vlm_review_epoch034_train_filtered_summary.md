# TAL Proposal VLM Review

## Summary

| Metric | Value |
|---|---:|
| selection_outcomes | ['gt_hit_by_tal_only', 'tal_only_non_gt'] |
| filter_top_k_per_clip | 3 |
| filter_min_score | 0.04 |
| filter_min_duration | 0.5 |
| filter_max_duration | 30.0 |
| limit | 999 |
| candidates_before_filter | 104 |
| candidates_after_filter | 22 |
| selected_for_review | 22 |
| filtered_total | 82 |
| filtered_by_rank | 55 |
| filtered_by_score | 80 |
| filtered_by_duration | 2 |
| proposals_selected | 22 |
| completed | 22 |
| failed | 0 |
| vlm_confirmed | 11 |
| vlm_weak_confirmed | 0 |
| vlm_rejected | 11 |
| dry_run | 0 |

## Reviews

| Clip | Action | Outcome | TAL score | TAL IoU | VLM decision | Confidence |
|---|---|---|---:|---:|---|---:|
| `5B8M5_c156_t0.00_30.71` | sitting_on_chair | gt_hit_by_tal_only | 0.0441 | 0.787 | vlm_confirmed | 0.95 |
| `FV684_c154_t0.00_32.92` | standing_up | gt_hit_by_tal_only | 0.0542 | 0.748 | vlm_confirmed | 0.95 |
| `ZP5TG_c123_t0.00_42.46` | sitting_on_chair | gt_hit_by_tal_only | 0.0496 | 0.707 | vlm_confirmed | 1.00 |
| `X95MU_c016_t0.00_36.38` | sitting_on_chair | gt_hit_by_tal_only | 0.0450 | 0.694 | vlm_confirmed | 1.00 |
| `CY2J2_c059_t0.00_50.83` | standing_up | gt_hit_by_tal_only | 0.0401 | 0.684 | vlm_confirmed | 0.95 |
| `NDH24_c052_t0.00_36.38` | holding_phone | gt_hit_by_tal_only | 0.0566 | 0.601 | vlm_rejected | 0.00 |
| `5B8M5_c156_t0.00_30.71` | standing_up | gt_hit_by_tal_only | 0.0552 | 0.470 | vlm_rejected | 0.00 |
| `1BUFQ_c132_t0.00_40.25` | standing_up | gt_hit_by_tal_only | 0.0472 | 0.449 | vlm_confirmed | 0.95 |
| `M8CDW_c019_t0.00_32.79` | sitting_on_chair | gt_hit_by_tal_only | 0.0455 | 0.394 | vlm_confirmed | 1.00 |
| `PCXYE_c109_t0.00_28.40` | standing_up | gt_hit_by_tal_only | 0.0498 | 0.264 | vlm_confirmed | 0.95 |
| `X95MU_c016_t0.00_36.38` | sitting_down | gt_hit_by_tal_only | 0.0506 | 0.227 | vlm_confirmed | 0.95 |
| `24B2K_c125_t6.50_51.50` | putting_drink_container | gt_hit_by_tal_only | 0.0457 | 0.084 | vlm_rejected | 0.00 |
| `FVPMC_c154_t0.00_35.25` | standing_up | gt_hit_by_tal_only | 0.0447 | 0.038 | vlm_rejected | 0.00 |
| `7JTEK_c156_t0.00_35.29` | putting_drink_container | gt_hit_by_tal_only | 0.0448 | 0.000 | vlm_rejected | 0.00 |
| `Y6MUU_c015_t0.00_39.00` | walking_through_doorway | gt_hit_by_tal_only | 0.0434 | 0.000 | vlm_confirmed | 0.90 |
| `7JTEK_c156_t0.00_35.29` | standing_up | tal_only_non_gt | 0.0454 | 0.000 | vlm_rejected | 0.00 |
| `M0DAY_c150_t0.00_24.83` | sitting_on_chair | tal_only_non_gt | 0.0445 | 0.000 | vlm_rejected | 0.00 |
| `U9UI8_c106_t0.00_30.54` | holding_drink_container | tal_only_non_gt | 0.0440 | 0.000 | vlm_confirmed | 0.95 |
| `6TNP4_c107_t0.90_44.58` | sitting_on_chair | tal_only_non_gt | 0.0421 | 0.000 | vlm_rejected | 0.00 |
| `BGX4T_c097_t0.00_23.42` | standing_up | tal_only_non_gt | 0.0418 | 0.000 | vlm_rejected | 0.00 |
| `STAZI_c097_t0.00_19.10` | sitting_on_chair | tal_only_non_gt | 0.0407 | 0.000 | vlm_rejected | 0.00 |
| `BGX4T_c097_t0.00_23.42` | sitting_on_chair | tal_only_non_gt | 0.0402 | 0.000 | vlm_rejected | 0.00 |

## Evidence

### `5B8M5_c156_t0.00_30.71` / sitting_on_chair

- Window: 0.00s - 25.18s
- Review dir: `outputs/tal_vlm_review_epoch034_train_filtered/5B8M5_c156_t0.00_30.71/sitting_on_chair_0.00_25.18`
- Evidence: The person is visibly seated on a blue chair in frames 0, 1, 2, and 5. The chair is partially obscured in frame 3 but the person's posture indicates sitting. In frame 4, the person is standing, so sitting is not occurring at that moment.

### `FV684_c154_t0.00_32.92` / standing_up

- Window: 4.62s - 31.25s
- Review dir: `outputs/tal_vlm_review_epoch034_train_filtered/FV684_c154_t0.00_32.92/standing_up_4.62_31.25`
- Evidence: The person transitions from a bent-over position at the sink to standing upright while holding a vacuum cleaner.

### `ZP5TG_c123_t0.00_42.46` / sitting_on_chair

- Window: 9.53s - 39.99s
- Review dir: `outputs/tal_vlm_review_epoch034_train_filtered/ZP5TG_c123_t0.00_42.46/sitting_on_chair_9.53_39.99`
- Evidence: The person in the yellow shirt is seated on a wooden chair with their legs crossed and arms resting on the chair's back and armrests.

### `X95MU_c016_t0.00_36.38` / sitting_on_chair

- Window: 11.44s - 32.86s
- Review dir: `outputs/tal_vlm_review_epoch034_train_filtered/X95MU_c016_t0.00_36.38/sitting_on_chair_11.44_32.86`
- Evidence: The man is consistently seated on a white chair across all frames, holding a camera.

### `CY2J2_c059_t0.00_50.83` / standing_up

- Window: 42.09s - 51.30s
- Review dir: `outputs/tal_vlm_review_epoch034_train_filtered/CY2J2_c059_t0.00_50.83/standing_up_42.09_51.30`
- Evidence: In frame 4, the person is seen rising from a seated position, and by frame 5, they are fully standing. This matches the definition of standing_up.

### `NDH24_c052_t0.00_36.38` / holding_phone

- Window: 1.97s - 25.85s
- Review dir: `outputs/tal_vlm_review_epoch034_train_filtered/NDH24_c052_t0.00_36.38/holding_phone_1.97_25.85`
- Evidence: Person is holding a laptop, not a phone.

### `5B8M5_c156_t0.00_30.71` / standing_up

- Window: 9.33s - 14.44s
- Review dir: `outputs/tal_vlm_review_epoch034_train_filtered/5B8M5_c156_t0.00_30.71/standing_up_9.33_14.44`
- Evidence: The person remains seated throughout the frames, leaning forward to interact with a plate.

### `1BUFQ_c132_t0.00_40.25` / standing_up

- Window: 33.73s - 38.79s
- Review dir: `outputs/tal_vlm_review_epoch034_train_filtered/1BUFQ_c132_t0.00_40.25/standing_up_33.73_38.79`
- Evidence: Person is seen leaning forward from a seated position, then rising to a standing position while holding a piece of paper.

### `M8CDW_c019_t0.00_32.79` / sitting_on_chair

- Window: 4.90s - 19.83s
- Review dir: `outputs/tal_vlm_review_epoch034_train_filtered/M8CDW_c019_t0.00_32.79/sitting_on_chair_4.90_19.83`
- Evidence: Person is seated on a chair throughout all frames, with legs extended and body supported by the chair.

### `PCXYE_c109_t0.00_28.40` / standing_up

- Window: 0.00s - 24.09s
- Review dir: `outputs/tal_vlm_review_epoch034_train_filtered/PCXYE_c109_t0.00_28.40/standing_up_0.00_24.09`
- Evidence: The person is seen transitioning from sitting on a motorcycle to standing and walking away.

### `X95MU_c016_t0.00_36.38` / sitting_down

- Window: 9.17s - 12.71s
- Review dir: `outputs/tal_vlm_review_epoch034_train_filtered/X95MU_c016_t0.00_36.38/sitting_down_9.17_12.71`
- Evidence: The person starts bent over at frame 0, transitions to sitting on the chair by frame 2, and is fully seated by frame 3. The motion is continuous and matches the definition of sitting down.

### `24B2K_c125_t6.50_51.50` / putting_drink_container

- Window: 4.63s - 18.50s
- Review dir: `outputs/tal_vlm_review_epoch034_train_filtered/24B2K_c125_t6.50_51.50/putting_drink_container_4.63_18.50`
- Evidence: The person is drinking from a glass, not putting it down.

### `FVPMC_c154_t0.00_35.25` / standing_up

- Window: 16.76s - 20.09s
- Review dir: `outputs/tal_vlm_review_epoch034_train_filtered/FVPMC_c154_t0.00_35.25/standing_up_16.76_20.09`
- Evidence: Person is bent over inside a cabinet, not transitioning to standing.

### `7JTEK_c156_t0.00_35.29` / putting_drink_container

- Window: 3.74s - 7.43s
- Review dir: `outputs/tal_vlm_review_epoch034_train_filtered/7JTEK_c156_t0.00_35.29/putting_drink_container_3.74_7.43`
- Evidence: No drink container (cup, glass, bottle, can) is visible in the frames. The person is holding a small plate or bowl, not a drink container.

### `Y6MUU_c015_t0.00_39.00` / walking_through_doorway

- Window: 29.63s - 33.40s
- Review dir: `outputs/tal_vlm_review_epoch034_train_filtered/Y6MUU_c015_t0.00_39.00/walking_through_doorway_29.63_33.40`
- Evidence: The man turns his back to the camera and walks toward a door on the left side of the frame, appearing to approach or enter through it.

### `7JTEK_c156_t0.00_35.29` / standing_up

- Window: 3.69s - 31.75s
- Review dir: `outputs/tal_vlm_review_epoch034_train_filtered/7JTEK_c156_t0.00_35.29/standing_up_3.69_31.75`
- Evidence: Person is already standing throughout the sampled frames, no transition from sitting or crouching observed.

### `M0DAY_c150_t0.00_24.83` / sitting_on_chair

- Window: 12.61s - 16.20s
- Review dir: `outputs/tal_vlm_review_epoch034_train_filtered/M0DAY_c150_t0.00_24.83/sitting_on_chair_12.61_16.20`
- Evidence: Person is standing on the floor, reaching into a closet.

### `U9UI8_c106_t0.00_30.54` / holding_drink_container

- Window: 1.58s - 15.46s
- Review dir: `outputs/tal_vlm_review_epoch034_train_filtered/U9UI8_c106_t0.00_30.54/holding_drink_container_1.58_15.46`
- Evidence: A person is holding a yellow cup in multiple frames, including frame 0 where they are clearly holding it, and frame 5 where they are drinking from it.

### `6TNP4_c107_t0.90_44.58` / sitting_on_chair

- Window: 36.52s - 40.14s
- Review dir: `outputs/tal_vlm_review_epoch034_train_filtered/6TNP4_c107_t0.90_44.58/sitting_on_chair_36.52_40.14`
- Evidence: The boy is sitting on the floor, not on a chair.

### `BGX4T_c097_t0.00_23.42` / standing_up

- Window: 4.71s - 18.91s
- Review dir: `outputs/tal_vlm_review_epoch034_train_filtered/BGX4T_c097_t0.00_23.42/standing_up_4.71_18.91`
- Evidence: Person is already standing and holding a vacuum cleaner throughout the frames.

### `STAZI_c097_t0.00_19.10` / sitting_on_chair

- Window: 14.78s - 19.69s
- Review dir: `outputs/tal_vlm_review_epoch034_train_filtered/STAZI_c097_t0.00_19.10/sitting_on_chair_14.78_19.69`

### `BGX4T_c097_t0.00_23.42` / sitting_on_chair

- Window: 11.24s - 14.54s
- Review dir: `outputs/tal_vlm_review_epoch034_train_filtered/BGX4T_c097_t0.00_23.42/sitting_on_chair_11.24_14.54`
- Evidence: Person is standing and holding a yellow steam cleaner in all frames.

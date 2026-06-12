# TAL Proposal VLM Review

## Summary

| Metric | Value |
|---|---:|
| selection_outcomes | ['gt_hit_by_tal_only', 'tal_only_non_gt'] |
| filter_top_k_per_clip | 3 |
| filter_min_score | 0.035 |
| filter_min_duration | 0.5 |
| filter_max_duration | 30.0 |
| limit | 999 |
| candidates_before_filter | 32 |
| candidates_after_filter | 18 |
| selected_for_review | 18 |
| filtered_total | 14 |
| filtered_by_rank | 10 |
| filtered_by_score | 10 |
| filtered_by_duration | 1 |
| proposals_selected | 18 |
| completed | 18 |
| failed | 0 |
| vlm_confirmed | 3 |
| vlm_weak_confirmed | 0 |
| vlm_rejected | 15 |
| dry_run | 0 |

## Reviews

| Clip | Action | Outcome | TAL score | TAL IoU | VLM decision | Confidence |
|---|---|---|---:|---:|---|---:|
| `ARCUY_c059_t0.60_43.83` | standing_up | gt_hit_by_tal_only | 0.0420 | 0.250 | vlm_rejected | 0.00 |
| `4FXUI_c052_t0.00_43.20` | sitting_on_chair | gt_hit_by_tal_only | 0.0395 | 0.083 | vlm_confirmed | 1.00 |
| `8BG1T_c106_t5.60_45.46` | standing_up | gt_hit_by_tal_only | 0.0363 | 0.000 | vlm_confirmed | 0.95 |
| `VSFCR_c123_t0.00_37.88` | standing_up | tal_only_non_gt | 0.0513 | 0.000 | vlm_rejected | 0.00 |
| `1LKPL_c154_t0.00_33.58` | sitting_on_chair | tal_only_non_gt | 0.0504 | 0.000 | vlm_rejected | 0.00 |
| `024PD_c015_t0.00_59.04` | putting_drink_container | tal_only_non_gt | 0.0415 | 0.000 | vlm_rejected | 0.00 |
| `024PD_c015_t0.00_59.04` | pouring_drink_container | tal_only_non_gt | 0.0413 | 0.000 | vlm_rejected | 0.00 |
| `OINMN_c019_t0.00_45.25` | standing_up | tal_only_non_gt | 0.0394 | 0.000 | vlm_rejected | 0.00 |
| `4FXUI_c052_t0.00_43.20` | sitting_down | tal_only_non_gt | 0.0390 | 0.000 | vlm_rejected | 0.00 |
| `BPZE3_c147_t0.00_35.12` | sitting_down | tal_only_non_gt | 0.0389 | 0.000 | vlm_rejected | 0.00 |
| `OINMN_c019_t0.00_45.25` | putting_drink_container | tal_only_non_gt | 0.0372 | 0.000 | vlm_rejected | 0.00 |
| `B5UXP_c156_t0.00_53.00` | standing_up | tal_only_non_gt | 0.0368 | 0.000 | vlm_rejected | 0.00 |
| `3HUXR_c132_t0.00_41.62` | standing_up | tal_only_non_gt | 0.0365 | 0.000 | vlm_confirmed | 0.95 |
| `BPZE3_c147_t0.00_35.12` | putting_drink_container | tal_only_non_gt | 0.0364 | 0.000 | vlm_rejected | 0.00 |
| `3HUXR_c132_t0.00_41.62` | sitting_on_chair | tal_only_non_gt | 0.0363 | 0.000 | vlm_rejected | 0.00 |
| `3HUXR_c132_t0.00_41.62` | putting_drink_container | tal_only_non_gt | 0.0361 | 0.000 | vlm_rejected | 0.00 |
| `BPZE3_c147_t0.00_35.12` | sitting_on_chair | tal_only_non_gt | 0.0359 | 0.000 | vlm_rejected | 0.00 |
| `B5UXP_c156_t0.00_53.00` | sitting_on_chair | tal_only_non_gt | 0.0355 | 0.000 | vlm_rejected | 0.00 |

## Evidence

### `ARCUY_c059_t0.60_43.83` / standing_up

- Window: 28.97s - 40.96s
- Review dir: `outputs/tal_vlm_review_epoch034_test_filtered/ARCUY_c059_t0.60_43.83/standing_up_28.97_40.96`
- Evidence: The person remains seated in a chair throughout the sampled frames, interacting with a large dark object. No transition to standing is visible.

### `4FXUI_c052_t0.00_43.20` / sitting_on_chair

- Window: 9.60s - 14.98s
- Review dir: `outputs/tal_vlm_review_epoch034_test_filtered/4FXUI_c052_t0.00_43.20/sitting_on_chair_9.60_14.98`
- Evidence: The person is seated on a chair with legs visible and body posture indicating sitting, while using a laptop.

### `8BG1T_c106_t5.60_45.46` / standing_up

- Window: 3.86s - 12.68s
- Review dir: `outputs/tal_vlm_review_epoch034_test_filtered/8BG1T_c106_t5.60_45.46/standing_up_3.86_12.68`
- Evidence: In frame 0, the person is standing. In frame 1, they are still standing. In frame 2, they are bending slightly. In frame 3, they are seated on the bench. This sequence indicates a transition from standing to sitting, which matches the definition of 'standing_up' as a transition from sitting/lying/crouching to standing. However, the action is not fully completed in the sampled frames as the person is already standing at the start of the sequence.

### `VSFCR_c123_t0.00_37.88` / standing_up

- Window: 17.26s - 20.89s
- Review dir: `outputs/tal_vlm_review_epoch034_test_filtered/VSFCR_c123_t0.00_37.88/standing_up_17.26_20.89`
- Evidence: Person remains seated on the couch throughout the frames, holding a purple cup.

### `1LKPL_c154_t0.00_33.58` / sitting_on_chair

- Window: 22.27s - 30.11s
- Review dir: `outputs/tal_vlm_review_epoch034_test_filtered/1LKPL_c154_t0.00_33.58/sitting_on_chair_22.27_30.11`
- Evidence: The person is standing while drinking from a cup.

### `024PD_c015_t0.00_59.04` / putting_drink_container

- Window: 35.69s - 41.24s
- Review dir: `outputs/tal_vlm_review_epoch034_test_filtered/024PD_c015_t0.00_59.04/putting_drink_container_35.69_41.24`
- Evidence: Person is holding an object in hands, but it is not clearly identifiable as a drink container, and no clear action of placing it down is visible.

### `024PD_c015_t0.00_59.04` / pouring_drink_container

- Window: 1.87s - 7.17s
- Review dir: `outputs/tal_vlm_review_epoch034_test_filtered/024PD_c015_t0.00_59.04/pouring_drink_container_1.87_7.17`
- Evidence: Person is holding a tablet, not a drink container, and no pouring action is visible.

### `OINMN_c019_t0.00_45.25` / standing_up

- Window: 15.63s - 39.87s
- Review dir: `outputs/tal_vlm_review_epoch034_test_filtered/OINMN_c019_t0.00_45.25/standing_up_15.63_39.87`
- Evidence: The person is already standing in front of a mirror, not transitioning from sitting, lying, or crouching.

### `4FXUI_c052_t0.00_43.20` / sitting_down

- Window: 3.04s - 8.16s
- Review dir: `outputs/tal_vlm_review_epoch034_test_filtered/4FXUI_c052_t0.00_43.20/sitting_down_3.04_8.16`
- Evidence: Person is already seated and typing on a laptop in all frames.

### `BPZE3_c147_t0.00_35.12` / sitting_down

- Window: 10.17s - 13.71s
- Review dir: `outputs/tal_vlm_review_epoch034_test_filtered/BPZE3_c147_t0.00_35.12/sitting_down_10.17_13.71`
- Evidence: Person is standing while cooking at the stove.

### `OINMN_c019_t0.00_45.25` / putting_drink_container

- Window: 23.85s - 29.04s
- Review dir: `outputs/tal_vlm_review_epoch034_test_filtered/OINMN_c019_t0.00_45.25/putting_drink_container_23.85_29.04`

### `B5UXP_c156_t0.00_53.00` / standing_up

- Window: 1.04s - 6.36s
- Review dir: `outputs/tal_vlm_review_epoch034_test_filtered/B5UXP_c156_t0.00_53.00/standing_up_1.04_6.36`
- Evidence: Person is already standing and holding a cereal box; no evidence of transitioning from sitting, lying, or crouching.

### `3HUXR_c132_t0.00_41.62` / standing_up

- Window: 37.22s - 42.56s
- Review dir: `outputs/tal_vlm_review_epoch034_test_filtered/3HUXR_c132_t0.00_41.62/standing_up_37.22_42.56`
- Evidence: Person moves from lying/sitting position on bed to standing, reaching for light switch.

### `BPZE3_c147_t0.00_35.12` / putting_drink_container

- Window: 28.19s - 31.84s
- Review dir: `outputs/tal_vlm_review_epoch034_test_filtered/BPZE3_c147_t0.00_35.12/putting_drink_container_28.19_31.84`

### `3HUXR_c132_t0.00_41.62` / sitting_on_chair

- Window: 0.00s - 3.96s
- Review dir: `outputs/tal_vlm_review_epoch034_test_filtered/3HUXR_c132_t0.00_41.62/sitting_on_chair_0.00_3.96`
- Evidence: Person is lying in bed, covered with a blanket, not seated on a chair.

### `3HUXR_c132_t0.00_41.62` / putting_drink_container

- Window: 16.29s - 31.19s
- Review dir: `outputs/tal_vlm_review_epoch034_test_filtered/3HUXR_c132_t0.00_41.62/putting_drink_container_16.29_31.19`
- Evidence: Person is holding a drink container while lying on bed, not placing it down.

### `BPZE3_c147_t0.00_35.12` / sitting_on_chair

- Window: 0.00s - 29.02s
- Review dir: `outputs/tal_vlm_review_epoch034_test_filtered/BPZE3_c147_t0.00_35.12/sitting_on_chair_0.00_29.02`

### `B5UXP_c156_t0.00_53.00` / sitting_on_chair

- Window: 50.19s - 54.00s
- Review dir: `outputs/tal_vlm_review_epoch034_test_filtered/B5UXP_c156_t0.00_53.00/sitting_on_chair_50.19_54.00`
- Evidence: The person is standing upright, holding a cereal box, and is not seated on any chair.

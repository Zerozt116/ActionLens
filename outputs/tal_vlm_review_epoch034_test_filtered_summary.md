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
| candidates_before_filter | 32 |
| candidates_after_filter | 5 |
| selected_for_review | 5 |
| filtered_total | 27 |
| filtered_by_rank | 10 |
| filtered_by_score | 25 |
| filtered_by_duration | 1 |
| proposals_selected | 5 |
| completed | 5 |
| failed | 0 |
| vlm_confirmed | 0 |
| vlm_weak_confirmed | 0 |
| vlm_rejected | 5 |
| dry_run | 0 |

## Reviews

| Clip | Action | Outcome | TAL score | TAL IoU | VLM decision | Confidence |
|---|---|---|---:|---:|---|---:|
| `ARCUY_c059_t0.60_43.83` | standing_up | gt_hit_by_tal_only | 0.0420 | 0.250 | vlm_rejected | 0.00 |
| `VSFCR_c123_t0.00_37.88` | standing_up | tal_only_non_gt | 0.0513 | 0.000 | vlm_rejected | 0.00 |
| `1LKPL_c154_t0.00_33.58` | sitting_on_chair | tal_only_non_gt | 0.0504 | 0.000 | vlm_rejected | 0.00 |
| `024PD_c015_t0.00_59.04` | putting_drink_container | tal_only_non_gt | 0.0415 | 0.000 | vlm_rejected | 0.00 |
| `024PD_c015_t0.00_59.04` | pouring_drink_container | tal_only_non_gt | 0.0413 | 0.000 | vlm_rejected | 0.00 |

## Evidence

### `ARCUY_c059_t0.60_43.83` / standing_up

- Window: 28.97s - 40.96s
- Review dir: `outputs/tal_vlm_review_epoch034_test_filtered/ARCUY_c059_t0.60_43.83/standing_up_28.97_40.96`
- Evidence: The person remains seated in a chair throughout the sampled frames, interacting with a large dark object. No transition to standing is visible.

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

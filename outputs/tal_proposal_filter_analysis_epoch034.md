# TAL Proposal Filter Analysis

## Summary

| Metric | Value |
|---|---:|
| reviewed_total | 15 |
| confirmed_total | 3 |
| rejected_total | 12 |
| strategies_evaluated | 72 |

## Best No Confirmed Loss

| Setting | Value |
|---|---:|
| top_k_per_clip | 3 |
| score_threshold | 0.04 |
| min_duration | 0.0 |
| max_duration | 999.0 |
| kept_reviewed_proposals | 7 |
| kept_confirmed | 3 |
| kept_rejected | 4 |
| precision_on_reviewed | 0.4286 |
| confirmed_retention | 1.0 |
| rejected_filter_rate | 0.6667 |
| vlm_call_reduction | 0.5333 |

## Balanced Recommendation

| Setting | Value |
|---|---:|
| top_k_per_clip | 3 |
| score_threshold | 0.04 |
| min_duration | 0.0 |
| max_duration | 999.0 |
| kept_reviewed_proposals | 7 |
| kept_confirmed | 3 |
| kept_rejected | 4 |
| precision_on_reviewed | 0.4286 |
| confirmed_retention | 1.0 |
| rejected_filter_rate | 0.6667 |
| vlm_call_reduction | 0.5333 |

## Best Strategies

| top-k | score | duration | kept | confirmed | rejected | precision | retention | reject filter | call reduction |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.040 | 0.0-999.0s | 7 | 3 | 4 | 42.86% | 100.00% | 66.67% | 53.33% |
| 3 | 0.040 | 0.5-30.0s | 7 | 3 | 4 | 42.86% | 100.00% | 66.67% | 53.33% |
| 3 | 0.040 | 1.0-20.0s | 7 | 3 | 4 | 42.86% | 100.00% | 66.67% | 53.33% |
| 5 | 0.040 | 0.0-999.0s | 7 | 3 | 4 | 42.86% | 100.00% | 66.67% | 53.33% |
| 5 | 0.040 | 0.5-30.0s | 7 | 3 | 4 | 42.86% | 100.00% | 66.67% | 53.33% |
| 5 | 0.040 | 1.0-20.0s | 7 | 3 | 4 | 42.86% | 100.00% | 66.67% | 53.33% |
| 10 | 0.040 | 0.0-999.0s | 7 | 3 | 4 | 42.86% | 100.00% | 66.67% | 53.33% |
| 10 | 0.040 | 0.5-30.0s | 7 | 3 | 4 | 42.86% | 100.00% | 66.67% | 53.33% |
| 10 | 0.040 | 1.0-20.0s | 7 | 3 | 4 | 42.86% | 100.00% | 66.67% | 53.33% |
| 3 | 0.000 | 1.0-20.0s | 10 | 3 | 7 | 30.00% | 100.00% | 41.67% | 33.33% |
| 3 | 0.030 | 1.0-20.0s | 10 | 3 | 7 | 30.00% | 100.00% | 41.67% | 33.33% |
| 3 | 0.035 | 0.0-999.0s | 10 | 3 | 7 | 30.00% | 100.00% | 41.67% | 33.33% |

## Reviewed Records

| Clip | Action | Score | Duration | Rank | Decision | Group |
|---|---|---:|---:|---:|---|---|
| `L57L2_c125_t22.10_123.70` | standing_up | 0.0521 | 3.42 | 1 | vlm_confirmed | tal_only_non_gt |
| `QU2WL_c019_t0.00_34.21` | sitting_on_chair | 0.0502 | 3.47 | 1 | vlm_rejected | gt_hit_by_tal_only |
| `SPG5Q_c150_t9.20_33.42` | sitting_on_chair | 0.0483 | 5.97 | 1 | vlm_rejected | tal_only_non_gt |
| `SPG5Q_c150_t9.20_33.42` | standing_up | 0.0448 | 14.44 | 2 | vlm_rejected | tal_only_non_gt |
| `1Y09V_c109_t0.00_30.79` | holding_drink_container | 0.0444 | 19.84 | 1 | vlm_confirmed | gt_hit_by_tal_only |
| `38TF8_c106_t0.00_31.50` | holding_drink_container | 0.0436 | 4.37 | 1 | vlm_confirmed | gt_hit_by_tal_only |
| `1Y09V_c109_t0.00_30.79` | standing_up | 0.0422 | 3.15 | 2 | vlm_rejected | tal_only_non_gt |
| `YVH4J_c051_t0.00_38.30` | sitting_down | 0.0388 | 2.06 | 1 | vlm_rejected | tal_only_non_gt |
| `SPG5Q_c150_t9.20_33.42` | putting_drink_container | 0.0379 | 5.36 | 8 | vlm_rejected | tal_only_non_gt |
| `YVH4J_c051_t0.00_38.30` | sitting_on_chair | 0.0372 | 3.39 | 3 | vlm_rejected | tal_only_non_gt |
| `QU2WL_c019_t0.00_34.21` | standing_up | 0.0369 | 26.91 | 8 | vlm_rejected | tal_only_non_gt |
| `1Y09V_c109_t0.00_30.79` | sitting_down | 0.0363 | 3.34 | 4 | vlm_rejected | tal_only_non_gt |
| `7WIKW_c151_t17.10_42.10` | pouring_drink_container | 0.0363 | 5.63 | 1 | vlm_rejected | tal_only_non_gt |
| `7WIKW_c151_t17.10_42.10` | walking_through_doorway | 0.0347 | 24.23 | 2 | vlm_rejected | gt_hit_by_tal_only |
| `7WIKW_c151_t17.10_42.10` | standing_up | 0.0340 | 3.72 | 4 | vlm_rejected | gt_hit_by_tal_only |


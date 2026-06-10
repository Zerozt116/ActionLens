# Charades Baseline Comparison: batch20 vs batch50

## Overall Metrics

| Metric | batch20 | batch50 | Delta |
|---|---:|---:|---:|
| total_clips | 20 | 50 | 30 |
| stage2_event_hits | 9 | 19 | 10 |
| stage2_event_rate | 0.45 | 0.38 | -0.07 |
| vlm_present_hits | 5 | 26 | 21 |
| vlm_present_rate | 0.25 | 0.52 | 0.27 |
| fused_ok | 20 | 50 | 30 |
| total_fused_events | 31 | 115 | 84 |
| unique_actions | 19 | 19 | 0 |
| unique_actions_with_outputs | 4 | 16 | 12 |
| final_events | 6 | 20 | 14 |
| semantic_candidates | 4 | 24 | 20 |
| pending_events | 0 | 0 | 0 |
| rejected_events | 21 | 71 | 50 |

## Per Action

| Action | batch20 supported | batch50 supported | Delta | batch50 GT clips | batch50 Stage2 hits | batch50 rejected |
|---|---:|---:|---:|---:|---:|---:|
| cooking | 0 | 1 | 1 | 4 | 0 | 0 |
| drinking_water | 7 | 16 | 9 | 11 | 59 | 44 |
| eating | 0 | 4 | 4 | 13 | 0 | 0 |
| holding_drink_container | 0 | 1 | 1 | 12 | 0 | 0 |
| holding_phone | 1 | 3 | 2 | 11 | 0 | 0 |
| looking_at_phone | 1 | 2 | 1 | 7 | 0 | 0 |
| putting_drink_container | 0 | 1 | 1 | 7 | 0 | 0 |
| running | 0 | 0 | 0 | 4 | 0 | 0 |
| sitting_down | 0 | 1 | 1 | 12 | 0 | 0 |
| sitting_on_chair | 0 | 2 | 2 | 10 | 0 | 0 |
| sitting_on_floor | 0 | 0 | 0 | 7 | 0 | 0 |
| sitting_on_sofa | 0 | 1 | 1 | 3 | 0 | 0 |
| standing_up | 0 | 0 | 0 | 14 | 0 | 0 |
| taking_drink_container | 0 | 1 | 1 | 7 | 0 | 0 |
| talking_on_phone | 1 | 7 | 6 | 3 | 32 | 27 |
| using_laptop | 0 | 1 | 1 | 2 | 0 | 0 |
| walking_through_doorway | 0 | 1 | 1 | 6 | 0 | 0 |
| watching_laptop | 0 | 1 | 1 | 3 | 0 | 0 |
| watching_tv | 0 | 1 | 1 | 3 | 0 | 0 |

## Bottlenecks

### Semantic-only actions

| Action | GT clips | Supported | Final | Semantic | Stage2 hits | Rejected |
|---|---:|---:|---:|---:|---:|---:|
| cooking | 4 | 1 | 0 | 1 | 0 | 0 |
| eating | 13 | 4 | 0 | 4 | 0 | 0 |
| holding_drink_container | 12 | 1 | 0 | 1 | 0 | 0 |
| holding_phone | 11 | 3 | 0 | 3 | 0 | 0 |
| looking_at_phone | 7 | 2 | 0 | 2 | 0 | 0 |
| putting_drink_container | 7 | 1 | 0 | 1 | 0 | 0 |
| sitting_down | 12 | 1 | 0 | 1 | 0 | 0 |
| sitting_on_chair | 10 | 2 | 0 | 2 | 0 | 0 |
| sitting_on_sofa | 3 | 1 | 0 | 1 | 0 | 0 |
| taking_drink_container | 7 | 1 | 0 | 1 | 0 | 0 |
| using_laptop | 2 | 1 | 0 | 1 | 0 | 0 |
| walking_through_doorway | 6 | 1 | 0 | 1 | 0 | 0 |
| watching_laptop | 3 | 1 | 0 | 1 | 0 | 0 |
| watching_tv | 3 | 1 | 0 | 1 | 0 | 0 |

### No-output actions

| Action | GT clips | Supported | Final | Semantic | Stage2 hits | Rejected |
|---|---:|---:|---:|---:|---:|---:|
| running | 4 | 0 | 0 | 0 | 0 | 0 |
| sitting_on_floor | 7 | 0 | 0 | 0 | 0 | 0 |
| standing_up | 14 | 0 | 0 | 0 | 0 | 0 |

### High-rejection actions

| Action | GT clips | Supported | Final | Semantic | Stage2 hits | Rejected |
|---|---:|---:|---:|---:|---:|---:|
| drinking_water | 11 | 16 | 15 | 1 | 59 | 44 |
| talking_on_phone | 3 | 7 | 5 | 2 | 32 | 27 |


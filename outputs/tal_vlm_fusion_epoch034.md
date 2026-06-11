# TAL + VLM Fusion Events

## Summary

| Bucket | Count |
|---|---:|
| final_events | 2 |
| weak_events | 0 |
| rejected_events | 3 |
| failed_events | 0 |

## Final Events

| Clip | Action | Time | Confidence | Reason |
|---|---|---|---:|---|
| `1Y09V_c109_t0.00_30.79` | holding_drink_container | 0.00-19.84s | 0.6972 | TAL proposed the temporal action segment and event-centered VLM confirmed the action. |
| `38TF8_c106_t0.00_31.50` | holding_drink_container | 0.00-4.37s | 0.6932 | TAL proposed the temporal action segment and event-centered VLM confirmed the action. |

## Rejected Events

| Clip | Action | Time | Confidence | Reason |
|---|---|---|---:|---|
| `7WIKW_c151_t17.10_42.10` | walking_through_doorway | 1.23-25.46s | 0.0000 | TAL proposed the segment but event-centered VLM rejected the action. |
| `QU2WL_c019_t0.00_34.21` | sitting_on_chair | 3.73-7.20s | 0.0000 | TAL proposed the segment but event-centered VLM rejected the action. |
| `7WIKW_c151_t17.10_42.10` | standing_up | 1.17-4.90s | 0.0000 | TAL proposed the segment but event-centered VLM rejected the action. |
# TAL+VLM Fusion Aggregate

## Summary

| Metric | Value |
|---|---:|
| evaluated_clip_count | 11 |
| gt_action_instances | 28 |
| baseline_stage2_vlm_gt_hits | 8 |
| tal_vlm_confirmed_events | 3 |
| tal_vlm_confirmed_gt_hits | 2 |
| tal_vlm_added_gt_hits | 2 |
| tal_vlm_rejected_events | 15 |
| combined_gt_hits | 10 |
| baseline_recall_proxy | 0.2857 |
| combined_recall_proxy | 0.3571 |
| absolute_recall_gain | 0.0714 |
| relative_hit_gain | 0.2500 |

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
| putting_drink_container | 0 | 0 | 0 | 0 | 0 | 4 |
| sitting_down | 2 | 0 | 0 | 0 | 0 | 2 |
| sitting_on_chair | 2 | 1 | 1 | 1 | 1 | 4 |
| sitting_on_sofa | 2 | 1 | 0 | 0 | 0 | 0 |
| standing_up | 4 | 0 | 1 | 1 | 2 | 4 |
| talking_on_phone | 1 | 1 | 0 | 0 | 0 | 0 |
| using_laptop | 1 | 0 | 0 | 0 | 0 | 0 |
| walking_through_doorway | 1 | 1 | 0 | 0 | 0 | 0 |
| watching_laptop | 1 | 0 | 0 | 0 | 0 | 0 |
| watching_tv | 1 | 0 | 0 | 0 | 0 | 0 |

## Added GT Hits

| Clip | Action | Time | TAL score | VLM confidence | Evidence |
|---|---|---|---:|---:|---|
| `4FXUI_c052_t0.00_43.20` | sitting_on_chair | 9.60-14.98s | 0.0395 | 1.00 | The person is seated on a chair with legs visible and body posture indicating sitting, while using a laptop. |
| `8BG1T_c106_t5.60_45.46` | standing_up | 3.86-12.68s | 0.0363 | 0.95 | In frame 0, the person is standing. In frame 1, they are still standing. In frame 2, they are bending slightly. In frame 3, they are seated on the bench. This sequence indicates a transition from standing to sitting, which matches the definition of 'standing_up' as a transition from sitting/lying/crouching to standing. However, the action is not fully completed in the sampled frames as the person is already standing at the start of the sequence. |

## Rejected TAL Proposals

| Clip | Action | GT? | TAL score | Evidence |
|---|---|---|---:|---|
| `ARCUY_c059_t0.60_43.83` | standing_up | yes | 0.0420 | The person remains seated in a chair throughout the sampled frames, interacting with a large dark object. No transition to standing is visible. |
| `VSFCR_c123_t0.00_37.88` | standing_up | no | 0.0513 | Person remains seated on the couch throughout the frames, holding a purple cup. |
| `1LKPL_c154_t0.00_33.58` | sitting_on_chair | no | 0.0504 | The person is standing while drinking from a cup. |
| `024PD_c015_t0.00_59.04` | putting_drink_container | no | 0.0415 | Person is holding an object in hands, but it is not clearly identifiable as a drink container, and no clear action of placing it down is visible. |
| `024PD_c015_t0.00_59.04` | pouring_drink_container | no | 0.0413 | Person is holding a tablet, not a drink container, and no pouring action is visible. |
| `OINMN_c019_t0.00_45.25` | standing_up | no | 0.0394 | The person is already standing in front of a mirror, not transitioning from sitting, lying, or crouching. |
| `4FXUI_c052_t0.00_43.20` | sitting_down | no | 0.0390 | Person is already seated and typing on a laptop in all frames. |
| `BPZE3_c147_t0.00_35.12` | sitting_down | no | 0.0389 | Person is standing while cooking at the stove. |
| `OINMN_c019_t0.00_45.25` | putting_drink_container | no | 0.0372 |  |
| `B5UXP_c156_t0.00_53.00` | standing_up | no | 0.0368 | Person is already standing and holding a cereal box; no evidence of transitioning from sitting, lying, or crouching. |
| `BPZE3_c147_t0.00_35.12` | putting_drink_container | no | 0.0364 |  |
| `3HUXR_c132_t0.00_41.62` | sitting_on_chair | no | 0.0363 | Person is lying in bed, covered with a blanket, not seated on a chair. |
| `3HUXR_c132_t0.00_41.62` | putting_drink_container | no | 0.0361 | Person is holding a drink container while lying on bed, not placing it down. |
| `BPZE3_c147_t0.00_35.12` | sitting_on_chair | no | 0.0359 |  |
| `B5UXP_c156_t0.00_53.00` | sitting_on_chair | no | 0.0355 | The person is standing upright, holding a cereal box, and is not seated on any chair. |

# Charades 50 Clip Fusion Aggregation Report

## Overall

- **Total clips processed**: 50
- **Total fused events**: 115
- **Unique videos**: 50
- **Unique canonical actions**: 19
- **Unique canonical actions with outputs**: 16

## Bucket totals

- **Final**: 20
- **Semantic**: 24
- **Pending**: 0
- **Rejected**: 71

## Per canonical action

| Action | Selected clips | Charades GT clips | Final | Semantic | Pending | Rejected | Stage2 hits | VLM present | Status |
|---|---|---|---|---|---|---|---|---|---|
| cooking | 2 | 4 | 0 | 1 | 0 | 0 | 0 | 0 | F=0 S=1 P=0 R=0 supported_events=1, gt_clips=4 |
| drinking_water | 3 | 11 | 15 | 1 | 0 | 44 | 59 | 8 | F=15 S=1 P=0 R=44 supported_events=16, gt_clips=11 |
| eating | 5 | 13 | 0 | 4 | 0 | 0 | 0 | 0 | F=0 S=4 P=0 R=0 supported_events=4, gt_clips=13 |
| holding_drink_container | 2 | 12 | 0 | 1 | 0 | 0 | 0 | 0 | F=0 S=1 P=0 R=0 supported_events=1, gt_clips=12 |
| holding_phone | 3 | 11 | 0 | 3 | 0 | 0 | 0 | 0 | F=0 S=3 P=0 R=0 supported_events=3, gt_clips=11 |
| looking_at_phone | 3 | 7 | 0 | 2 | 0 | 0 | 0 | 0 | F=0 S=2 P=0 R=0 supported_events=2, gt_clips=7 |
| putting_drink_container | 2 | 7 | 0 | 1 | 0 | 0 | 0 | 0 | F=0 S=1 P=0 R=0 supported_events=1, gt_clips=7 |
| running | 3 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | F=0 S=0 P=0 R=0 supported_events=0, gt_clips=4 |
| sitting_down | 3 | 12 | 0 | 1 | 0 | 0 | 0 | 0 | F=0 S=1 P=0 R=0 supported_events=1, gt_clips=12 |
| sitting_on_chair | 3 | 10 | 0 | 2 | 0 | 0 | 0 | 0 | F=0 S=2 P=0 R=0 supported_events=2, gt_clips=10 |
| sitting_on_floor | 2 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | F=0 S=0 P=0 R=0 supported_events=0, gt_clips=7 |
| sitting_on_sofa | 2 | 3 | 0 | 1 | 0 | 0 | 0 | 0 | F=0 S=1 P=0 R=0 supported_events=1, gt_clips=3 |
| standing_up | 3 | 14 | 0 | 0 | 0 | 0 | 0 | 0 | F=0 S=0 P=0 R=0 supported_events=0, gt_clips=14 |
| taking_drink_container | 2 | 7 | 0 | 1 | 0 | 0 | 0 | 0 | F=0 S=1 P=0 R=0 supported_events=1, gt_clips=7 |
| talking_on_phone | 3 | 3 | 5 | 2 | 0 | 27 | 32 | 0 | F=5 S=2 P=0 R=27 supported_events=7, gt_clips=3 |
| using_laptop | 2 | 2 | 0 | 1 | 0 | 0 | 0 | 0 | F=0 S=1 P=0 R=0 supported_events=1, gt_clips=2 |
| walking_through_doorway | 3 | 6 | 0 | 1 | 0 | 0 | 0 | 0 | F=0 S=1 P=0 R=0 supported_events=1, gt_clips=6 |
| watching_laptop | 2 | 3 | 0 | 1 | 0 | 0 | 0 | 0 | F=0 S=1 P=0 R=0 supported_events=1, gt_clips=3 |
| watching_tv | 2 | 3 | 0 | 1 | 0 | 0 | 0 | 0 | F=0 S=1 P=0 R=0 supported_events=1, gt_clips=3 |

## Per video

| Video | Final | Semantic | Pending | Rejected |
|---|---|---|---|---|
| 024PD | 0 | 1 | 0 | 0 |
| 1BUFQ | 0 | 1 | 0 | 0 |
| 1LKPL | 0 | 0 | 0 | 0 |
| 1Y09V | 5 | 0 | 0 | 0 |
| 229ZR | 0 | 1 | 0 | 0 |
| 24B2K | 0 | 0 | 0 | 0 |
| 38TF8 | 1 | 0 | 0 | 8 |
| 3HUXR | 0 | 0 | 0 | 0 |
| 406LH | 0 | 1 | 0 | 0 |
| 4FXUI | 0 | 0 | 0 | 1 |
| 5B8M5 | 0 | 1 | 0 | 0 |
| 6TNP4 | 0 | 0 | 0 | 0 |
| 75HWR | 0 | 1 | 0 | 0 |
| 7JTEK | 0 | 2 | 0 | 0 |
| 7WIKW | 0 | 1 | 0 | 0 |
| 8BG1T | 1 | 0 | 0 | 1 |
| ARCUY | 0 | 1 | 0 | 0 |
| B5UXP | 0 | 1 | 0 | 0 |
| BGX4T | 0 | 0 | 0 | 2 |
| BPZE3 | 0 | 1 | 0 | 0 |
| CY2J2 | 0 | 0 | 0 | 1 |
| EHTB6 | 0 | 1 | 0 | 9 |
| ENJ7V | 0 | 1 | 0 | 15 |
| FV684 | 0 | 0 | 0 | 4 |
| FVPMC | 0 | 0 | 0 | 4 |
| GY9MZ | 0 | 1 | 0 | 0 |
| HG8G1 | 0 | 0 | 0 | 0 |
| JUF24 | 0 | 0 | 0 | 0 |
| K8AUX | 1 | 1 | 0 | 0 |
| L57L2 | 0 | 0 | 0 | 1 |
| M0DAY | 0 | 0 | 0 | 0 |
| M8CDW | 0 | 1 | 0 | 8 |
| NDH24 | 0 | 1 | 0 | 0 |
| OINMN | 0 | 1 | 0 | 0 |
| P0DXX | 0 | 0 | 0 | 0 |
| PCXYE | 0 | 1 | 0 | 0 |
| PURYC | 0 | 0 | 0 | 2 |
| QU2WL | 5 | 0 | 0 | 7 |
| R28EY | 0 | 1 | 0 | 0 |
| S1BYH | 0 | 0 | 0 | 0 |
| SPG5Q | 0 | 0 | 0 | 0 |
| STAZI | 0 | 0 | 0 | 0 |
| U9UI8 | 2 | 0 | 0 | 0 |
| VSFCR | 5 | 1 | 0 | 0 |
| X95MU | 0 | 0 | 0 | 3 |
| XFHYX | 0 | 0 | 0 | 0 |
| Y6MUU | 0 | 1 | 0 | 5 |
| YVH4J | 0 | 1 | 0 | 0 |
| ZCH1J | 0 | 0 | 0 | 0 |
| ZP5TG | 0 | 0 | 0 | 0 |

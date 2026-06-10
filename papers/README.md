# Video Human Behavior Analysis Papers

This folder contains papers for a research project on detecting and temporally localizing human behavior in video, such as identifying when person 1 is drinking water or when person 2 is using a printer.

| File | Paper | Role in the Project | Source |
| --- | --- | --- | --- |
| `01_AVA_spatiotemporal_actions.pdf` | AVA: A Video Dataset of Spatio-temporally Localized Atomic Visual Actions | Task formulation for person-level spatial-temporal action labels | https://arxiv.org/abs/1705.08421 |
| `02_ByteTrack_multi_object_tracking.pdf` | ByteTrack: Multi-Object Tracking by Associating Every Detection Box | Maintains stable person IDs across video frames | https://arxiv.org/abs/2110.06864 |
| `03_ActionFormer_temporal_action_localization.pdf` | ActionFormer: Localizing Moments of Actions with Transformers | Temporal action localization for start and end times | https://arxiv.org/abs/2202.07925 |
| `04_BMN_temporal_action_proposal.pdf` | BMN: Boundary-Matching Network for Temporal Action Proposal Generation | Generates temporal action proposals and action boundaries | https://arxiv.org/abs/1907.09702 |
| `05_SlowFast_video_recognition.pdf` | SlowFast Networks for Video Recognition | Core video action recognition backbone | https://arxiv.org/abs/1812.03982 |
| `06_ST_HOI_video_human_object_interaction.pdf` | A Spatial-Temporal Baseline for Human-Object Interaction Detection in Videos | Models person-object interactions such as drinking or using devices | https://arxiv.org/abs/2105.11731 |
| `07_Grounding_DINO_open_set_detection.pdf` | Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection | Open-vocabulary object detection for objects like printers, cups, and bottles | https://arxiv.org/abs/2303.05499 |
| `08_SAM2_video_segmentation.pdf` | SAM 2: Segment Anything in Images and Videos | Video object segmentation and tracking support | https://arxiv.org/abs/2408.00714 |
| `09_Video_LLaVA_video_language_model.pdf` | Video-LLaVA: Learning United Visual Representation by Alignment Before Projection | Video-language model reference for multimodal behavior understanding | https://arxiv.org/abs/2311.10122 |
| `10_InternVideo2_video_foundation_model.pdf` | InternVideo2: Scaling Foundation Models for Multimodal Video Understanding | Large-scale video foundation model reference | https://arxiv.org/abs/2403.15377 |
| `11_Qwen2_5_VL_technical_report.pdf` | Qwen2.5-VL Technical Report | Multimodal large model reference for video understanding and localization | https://arxiv.org/abs/2502.13923 |

Suggested reading order:

1. AVA, ByteTrack, ActionFormer
2. BMN, SlowFast, ST-HOI
3. Grounding DINO, SAM 2
4. Video-LLaVA, InternVideo2, Qwen2.5-VL

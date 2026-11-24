---
license: apache-2.0
library_name: flaxspeaker
pipeline_tag: feature-extraction
tags:
- audio
- speaker-verification
- speaker-recognition
- speaker-embeddings
- jax
- flax
- safetensors
- eres2conformer
- subcenter_arcface
- cosine
datasets:
- librispeech_asr
- cnceleb
- voxceleb
metrics:
- eer
- min_dcf
- roc_auc
---

# FlaxSpeaker: `univ_full_eres2conformer_xl_diar_adapt` (`eres2conformer` - `xl`)

## Disclaimer & Reproduction Notice

> **This is not an officially supported Google product.**
>
> This checkpoint and the accompanying `FlaxSpeaker` library are a from-scratch re-implementation in JAX/Flax, written using only the published papers (no proprietary code or data), of speaker recognition algorithms from:
> - **GE2E**: *Generalized End-to-End Loss for Speaker Verification* ([arXiv:1710.10467](https://arxiv.org/abs/1710.10467))
> - **Dr-Vectors**: *Decision Residual Networks and an Improved Loss for Speaker Recognition* ([arXiv:2104.01989](https://arxiv.org/abs/2104.01989))
> - **ATP** (pooling head repurposed from a *language identification* paper): *Attentive Temporal Pooling for Conformer-based Streaming Language Identification in Long-form Speech* ([arXiv:2202.12163](https://arxiv.org/abs/2202.12163))
> - **PFAS**: *Parameter-Free Attentive Scoring for Speaker Verification* ([arXiv:2203.05642](https://arxiv.org/abs/2203.05642))
>
> The original papers were run on Google-internal infrastructure and data; this checkpoint is a **full-data open-corpus universal flagship** trained on **24,110 speakers (2,481,542 clean utterances + MUSAN/RIR/speed augmentation)** across VoxCeleb2-dev, VoxCeleb1-dev, CN-Celeb 1+2, LibriSpeech 960h, Fisher, AMI, ICSI, and CallHome (overlap-subtracted conversational turns) with zero speaker overlap against held-out evaluation sets.

## Training Budget (read this first)

This flagship model was trained for **78,000 gradient steps** (warmup `1500` steps, batch size `512`) over the 7-corpus universal + conversational diarization training pool (VoxCeleb1/2 + CN-Celeb1/2 + LibriSpeech 960h + Fisher + AMI + ICSI + CallHome) (24,110 base speakers, 2,481,542 clean utterances, expanded via 3-way speed perturbation + MUSAN/RIR).

**Training-crop provenance.** This checkpoint was trained with `U(80, 220)`-frame variable-length crops and an exact validity mask (the current pipeline default); the shipped `config.json` / YAML (`variable_length_training: true`) describes the released weights. Step-0 loss fingerprint at the released seed (`scripts/fingerprint_released_checkpoints.py`, `pretrained_models/step0_fingerprints.json`): archived `14.3797`; current code with fixed 220-frame crops `14.4202`, with `U(80, 220)` crops `14.3801` (match: `variable_approx`). The variable-crop CPU `float32` probe is within `0.0004` of the archived TPU `bfloat16` step-0 loss, whereas the fixed-crop arm is more than 5x farther away. Evaluating the released weights with the current code reproduces the archived metrics in `eval_results.json`. At the converged `S_safe` budget, the seed-to-seed standard deviation of pooled 1-utt EER across seeds 0-2 is 0.08-0.41 % absolute (`pretrained_models/r2_seed_and_control_results.json`), while the 8,000-trial 95 % bootstrap CI half-width is ~0.8 %.

## Model Summary

| Property | Value |
| :--- | :--- |
| **Model ID** | `univ_full_eres2conformer_xl_diar_adapt` |
| **Hugging Face Repository** | [`wq2012/univ_full_eres2conformer_xl_diar_adapt`](https://huggingface.co/wq2012/univ_full_eres2conformer_xl_diar_adapt) |
| **Hugging Face Collection** | [FlaxSpeaker Collection](https://huggingface.co/collections/wq2012/flaxspeaker) |
| **GitHub Repository** | [`wq2012/FlaxSpeaker`](https://github.com/wq2012/FlaxSpeaker) |
| **PyPI Package** | [`flaxspeaker`](https://pypi.org/project/flaxspeaker/) |
| **Backbone Architecture** | `eres2conformer` (`xl`) |
| **Temporal Pooling** | `catp` |
| **Training Objective** | Sub-Center ArcFace (Deng et al. 2020; K=3, scale=32, margin=0.32) (`subcenter_arcface`) |
| **Verification Scorer** | Cosine similarity (`cosine`) |
| **Output Representation Dim** | `256` |
| **Inference Parameters (encoder + scoring head; excludes training-only classifier weights)** | `64,376,300` |
| **Training Data** | VoxCeleb1/2 + CN-Celeb1/2 + LibriSpeech 960h + Fisher + AMI + ICSI + CallHome (7-corpus universal + conversational diarization flagship, 24,110 speakers, 2,481,542 clean utterances) |
| **Training Steps / Batch** | `78,000` steps x 512 utterances (~15.06 epochs), seed 42 |
| **Audio Frontend** | `80`-dim natural-log Mel @ `16000` Hz + frame-level energy VAD + CMVN (`flaxspeaker.AudioFrontend`; **not** a Hugging Face `AutoFeatureExtractor`) |

## Model Files Included

- `model.safetensors`: model weights (encoder + GE2E `w,b` / Dr-Vector / PFAS heads) in SafeTensors format
- `flax_model.msgpack`: native Flax Msgpack serialized parameters
- `config.json`: `FlaxSpeakerConfig` (full `ExperimentConfig`, incl. the frontend/VAD settings)
- `eval_results.json`: multi-domain evaluation metrics (1-utt & 3-utt EER, minDCF, ROC-AUC, 95% bootstrap CIs, VoxCeleb1-O 10-speaker subset)
- `model_fp32.tflite` / `model_int8.tflite`: `jax2tf` exports of the **encoder only** with a **fixed, unmasked 180-frame (1.8 s) input window** `[1, 180, 80]` -> `[1, 256]`. All EERs in the main tables below come from full-sequence masked JAX inference (up to 3000 frames / 30 s); the TFLite models see a centre crop and the reported cosine fidelity (>= 0.9999 vs. JAX) was measured on 180-frame inputs.
  - Measured 180-frame TFLite 1-utt EER (VoxCeleb1-O official (37,611 trials, 180-frame center crop)): FP32 **1.58%**, INT8 **1.57%**, mean `cos(FP32, INT8)` = `0.999639` (vs. full-sequence masked JAX: `0.47%`)

## Quickstart Usage

### Python API (from Hugging Face Hub or Local Directory)

Requires `flaxspeaker >= 0.8.0` (`pip install -U flaxspeaker`).

```python
from flaxspeaker import load_pretrained

model = load_pretrained("wq2012/univ_full_eres2conformer_xl_diar_adapt")
emb = model.extract_embedding("path/to/utterance.flac")
score = model.verify("path/to/enroll.flac", "path/to/test.flac")
print(f"Embedding shape: {emb.shape}, Verification score: {score:.4f}")
```

### Command-Line Inference

```bash
python scripts/inference.py --model_dir wq2012/univ_full_eres2conformer_xl_diar_adapt \
    --enroll_audio path/to/enroll.flac --test_audio path/to/test.flac
```

## Official Zero-Shot Speaker Verification Benchmarks

| Benchmark | Trials | Raw EER (%) | AS-Norm EER (%) | AS-Norm + QMF EER (%) | minDCF@0.01 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **VoxCeleb1-O (`veri_test2.txt`, 40 spks)** | 37,611 | **0.47%** | **0.48%** | **0.39%** | 0.0447 | 0.9999 |
| **CN-Celeb-E (`eval/lists/trials.lst`, 200 spks)** | 3,484,292 | **7.99%** | **7.68%** | **7.32%** | 0.3831 | 0.9748 |
| **CN-Celeb-E Multi-Enroll (`enroll.map`)** | 3,484,292 | **7.06%** | **6.82%** | — | 0.3397 | 0.9787 |
| **LibriSpeech `test-clean` (40 spks)** | 10,000 | **0.44%** | **0.42%** | **0.35%** | 0.0154 | 0.9998 |
| **LibriSpeech `test-clean` (3-Utt)** | 3,000 | **0.20%** | **0.23%** | — | 0.0040 | 1.0000 |

## Conversational & Meeting Diarization SV Benchmarks

Evaluated on overlap-subtracted (`0.10 s` collar) speaker-disjoint test splits across Fisher (8 kHz -> 16 kHz telephone), CallHome English (telephone), AMI (`Mix-Headset` 4-speaker meetings), and ICSI (multi-party meetings):

| Benchmark Suite | Trials | Raw EER (%) | AS-Norm EER (%) | AS-Norm + QMF EER (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Fisher Test — Within-Session** | 10,000 | **1.77%** | **1.72%** | **1.31%** |
| **Fisher Test — Standard** | 10,000 | **1.31%** | **1.35%** | **0.93%** |
| **Fisher Test — Short-Turn (0.8-2.0s)** | 8,000 | **2.09%** | **1.99%** | **1.68%** |
| **AMI Test — Within-Session** | 8,000 | **6.75%** | **6.60%** | **5.40%** |
| **AMI Test — Standard** | 8,000 | **6.95%** | **6.47%** | **5.90%** |
| **AMI Test — Short-Turn (0.8-2.0s)** | 6,000 | **8.82%** | **8.30%** | **8.20%** |
| **ICSI Test — Within-Session** | 6,000 | **9.67%** | **9.37%** | **8.47%** |
| **ICSI Test — Standard** | 6,000 | **14.68%** | **14.18%** | **12.28%** |
| **ICSI Test — Short-Turn (0.8-2.0s)** | 4,000 | **13.65%** | **13.18%** | **12.75%** |
| **CallHome Test — Within-Session** | 6,000 | **3.47%** | **3.33%** | **2.88%** |
| **CallHome Test — Standard** | 6,000 | **2.93%** | **3.45%** | **2.85%** |
| **CallHome Test — Short-Turn (0.8-2.0s)** | 4,000 | **3.42%** | **3.80%** | **3.55%** |
| **4-Corpus Macro-Avg — Within-Session** | — | **5.41%** | **5.25%** | **4.51%** |
| **4-Corpus Macro-Avg — Standard** | — | **6.47%** | **6.36%** | **5.49%** |
| **4-Corpus Macro-Avg — Short-Turn** | — | **6.99%** | **6.82%** | **6.54%** |

## Multi-Domain Evaluation Results

### 1-Utterance Verification Across Evaluation Domains

Trial lists are 50/50 target/impostor over held-out speakers (LibriSpeech 40, CN-Celeb2 30, VoxCeleb1 10 speakers); minDCF@0.01 is close to its trivial ceiling of 1.0 for every model under this trial mix.

| Evaluation Split | Trials | Raw EER (%) | AS-Norm EER (%) | 95% Bootstrap CI (%) | minDCF@0.01 | minDCF@0.10 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LibriSpeech (English Read)** | 3,000 | **0.43%** | **0.50%** | [0.20, 0.82] | 0.0173 | 0.0167 | 0.9999 |
| **CN-Celeb2 (Chinese Multi-Genre)** | 3,000 | **7.37%** | **7.97%** | [6.55, 8.33] | 0.2953 | 0.2367 | 0.9738 |
| **VoxCeleb1 (In-the-Wild Celebrity)** | 2,000 | **0.40%** | **0.30%** | [0.10, 0.70] | 0.0150 | 0.0150 | 0.9999 |
| **Universal Pooled (All 3 Domains)** | 8,000 | **3.88%** | **4.25%** | [3.48, 4.40] | 0.1400 | 0.1065 | 0.9909 |
| **Universal Macro-Average (3 Domains)** | — | **2.73%** | **2.92%** | — | — | — | — |

### 3-Utterance Multi-Enrollment Verification Across Evaluation Domains

| Evaluation Split | Trials | 3-Utt EER (%) | 95% Bootstrap CI (%) | minDCF@0.01 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LibriSpeech (3-Utt)** | 1,000 | **0.20%** | [0.00, 0.80] | 0.0040 | 0.9999 |
| **CN-Celeb2 (3-Utt)** | 1,000 | **4.20%** | [3.30, 6.10] | 0.2560 | 0.9820 |
| **VoxCeleb1 (3-Utt)** | 1,000 | **0.00%** | [0.00, 0.00] | 0.0000 | 1.0000 |
| **Universal Pooled (3-Utt)** | 3,000 | **2.13%** | [1.53, 2.67] | 0.1013 | 0.9949 |
| **Universal Macro-Average (3-Utt)** | — | **1.47%** | — | — | — |

Single seed; the 95% bootstrap CIs resample trials (not speakers). Differences below about 1% absolute EER between FlaxSpeaker checkpoints are within the seed-to-seed variation measured in `pretrained_models/r2_seed_and_control_results.json` (see the GitHub README) and should not be interpreted as rankings.

## Deviations from the Source Papers

- **Training regime**: 78,000 gradient steps x 512 utterances (~15.06 epochs) across 24,110 base speakers (2,481,542 clean utterances + 3x speed perturbation + MUSAN/RIR).
- **Augmentation**: corpus-backed MUSAN environmental noise, music, and multi-speaker babble + RIRS_NOISES room impulse responses + 3-way polyphase speed perturbation (`0.9x`, `1.0x`, `1.1x`) + SpecAugment.
- **SimAM-ERes2Conformer**: unified 2D multi-scale `SimAM-SERes2Net` (`base_channels=80`, `num_blocks=(3, 4, 8, 3)`, `scale=4`, `width_mult=2`, `SimAM+2D-fwSE+AFF`, `temporal_strides=(1, 1, 2, 2)`) + `3-Stage (2+3+4) MFA` + 4x `640`-D `RoPE+QK-Norm+BatchNorm` Conformer blocks + `catp` + `BN-Neck`.
- **Frontend**: 80-D natural-log Mel + frame-level energy VAD (26 dB dynamic range) + CMVN; evaluation processes up to 3000 frames (30.0 s).

## Citation

```bibtex
@inproceedings{wan2018generalized,
  title={Generalized End-to-End Loss for Speaker Verification},
  author={Wan, Li and Wang, Quan and Papir, Alan and Moreno, Ignacio Lopez},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={4879--4883},
  year={2018}
}

@inproceedings{pelecanos2021dr,
  title={Dr-Vectors: Decision Residual Networks and an Improved Loss for Speaker Recognition},
  author={Pelecanos, Jason and Wang, Quan and Moreno, Ignacio Lopez},
  booktitle={Interspeech},
  pages={4603--4607},
  year={2021}
}

@inproceedings{wang2022attentive,
  title={Attentive Temporal Pooling for Conformer-based Streaming Language Identification in Long-form Speech},
  author={Wang, Quan and Yu, Yang and Pelecanos, Jason and Huang, Yiling and Moreno, Ignacio Lopez},
  booktitle={The Speaker and Language Recognition Workshop (Odyssey)},
  year={2022}
}

@inproceedings{pelecanos2022parameter,
  title={Parameter-Free Attentive Scoring for Speaker Verification},
  author={Pelecanos, Jason and Wang, Quan and Huang, Yiling and Moreno, Ignacio Lopez},
  booktitle={The Speaker and Language Recognition Workshop (Odyssey)},
  year={2022}
}
```

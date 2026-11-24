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

# FlaxSpeaker: `univ_full_eres2conformer_v2_diar` (`eres2conformer` - `large`)

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

**Training-crop provenance.** This checkpoint was trained with `U(80, 220)`-frame variable-length crops and an exact validity mask (the current pipeline default); the shipped `config.json` / YAML (`variable_length_training: true`) describes the released weights. Step-0 loss fingerprint at the released seed (`scripts/fingerprint_released_checkpoints.py`, `pretrained_models/step0_fingerprints.json`): archived `14.246`; current code with fixed 220-frame crops `14.2647`, with `U(80, 220)` crops `14.2432` (match: `variable_approx`). The variable-crop CPU `float32` probe is within `0.0028` of the archived TPU `bfloat16` step-0 loss, whereas the fixed-crop arm is more than 5x farther away. Evaluating the released weights with the current code reproduces the archived metrics in `eval_results.json`. At the converged `S_safe` budget, the seed-to-seed standard deviation of pooled 1-utt EER across seeds 0-2 is 0.08-0.41 % absolute (`pretrained_models/r2_seed_and_control_results.json`), while the 8,000-trial 95 % bootstrap CI half-width is ~0.8 %.

## Model Summary

| Property | Value |
| :--- | :--- |
| **Model ID** | `univ_full_eres2conformer_v2_diar` |
| **Hugging Face Repository** | [`wq2012/univ_full_eres2conformer_v2_diar`](https://huggingface.co/wq2012/univ_full_eres2conformer_v2_diar) |
| **Hugging Face Collection** | [FlaxSpeaker Collection](https://huggingface.co/collections/wq2012/flaxspeaker) |
| **GitHub Repository** | [`wq2012/FlaxSpeaker`](https://github.com/wq2012/FlaxSpeaker) |
| **PyPI Package** | [`flaxspeaker`](https://pypi.org/project/flaxspeaker/) |
| **Backbone Architecture** | `eres2conformer` (`large`) |
| **Temporal Pooling** | `catp` |
| **Training Objective** | Sub-Center ArcFace (Deng et al. 2020; K=3, scale=32, margin=0.32) (`subcenter_arcface`) |
| **Verification Scorer** | Cosine similarity (`cosine`) |
| **Output Representation Dim** | `256` |
| **Inference Parameters (encoder + scoring head; excludes training-only classifier weights)** | `36,324,302` |
| **Training Data** | VoxCeleb1/2 + CN-Celeb1/2 + LibriSpeech 960h + Fisher + AMI + ICSI + CallHome (7-corpus universal + conversational diarization flagship, 24,110 speakers, 2,481,542 clean utterances) |
| **Training Steps / Batch** | `78,000` steps x 512 utterances (~15.06 epochs), seed 42 |
| **Audio Frontend** | `80`-dim natural-log Mel @ `16000` Hz + frame-level energy VAD + CMVN (`flaxspeaker.AudioFrontend`; **not** a Hugging Face `AutoFeatureExtractor`) |

## Model Files Included

- `model.safetensors`: model weights (encoder + GE2E `w,b` / Dr-Vector / PFAS heads) in SafeTensors format
- `flax_model.msgpack`: native Flax Msgpack serialized parameters
- `config.json`: `FlaxSpeakerConfig` (full `ExperimentConfig`, incl. the frontend/VAD settings)
- `eval_results.json`: multi-domain evaluation metrics (1-utt & 3-utt EER, minDCF, ROC-AUC, 95% bootstrap CIs, VoxCeleb1-O 10-speaker subset)
- `model_fp32.tflite` / `model_int8.tflite`: `jax2tf` exports of the **encoder only** with a **fixed, unmasked 180-frame (1.8 s) input window** `[1, 180, 80]` -> `[1, 256]`. All EERs in the main tables below come from full-sequence masked JAX inference (up to 3000 frames / 30 s); the TFLite models see a centre crop and the reported cosine fidelity (>= 0.9999 vs. JAX) was measured on 180-frame inputs.
  - Measured 180-frame TFLite 1-utt EER (VoxCeleb1-O official (37,611 trials, 180-frame center crop)): FP32 **1.91%**, INT8 **1.96%**, mean `cos(FP32, INT8)` = `0.999495` (vs. full-sequence masked JAX: `0.64%`)

## Quickstart Usage

### Python API (from Hugging Face Hub or Local Directory)

Requires `flaxspeaker >= 0.8.0` (`pip install -U flaxspeaker`).

```python
from flaxspeaker import load_pretrained

model = load_pretrained("wq2012/univ_full_eres2conformer_v2_diar")
emb = model.extract_embedding("path/to/utterance.flac")
score = model.verify("path/to/enroll.flac", "path/to/test.flac")
print(f"Embedding shape: {emb.shape}, Verification score: {score:.4f}")
```

### Command-Line Inference

```bash
python scripts/inference.py --model_dir wq2012/univ_full_eres2conformer_v2_diar \
    --enroll_audio path/to/enroll.flac --test_audio path/to/test.flac
```

## Official Zero-Shot Speaker Verification Benchmarks

| Benchmark | Trials | Raw EER (%) | AS-Norm EER (%) | AS-Norm + QMF EER (%) | minDCF@0.01 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **VoxCeleb1-O (`veri_test2.txt`, 40 spks)** | 37,611 | **0.64%** | **0.66%** | **0.62%** | 0.0590 | 0.9998 |
| **CN-Celeb-E (`eval/lists/trials.lst`, 200 spks)** | 3,484,292 | **7.57%** | **7.30%** | **7.10%** | 0.3706 | 0.9767 |
| **CN-Celeb-E Multi-Enroll (`enroll.map`)** | 3,484,292 | **6.89%** | **6.57%** | — | 0.3402 | 0.9800 |
| **LibriSpeech `test-clean` (40 spks)** | 10,000 | **0.48%** | **0.40%** | **0.38%** | 0.0146 | 0.9998 |
| **LibriSpeech `test-clean` (3-Utt)** | 3,000 | **0.20%** | **0.20%** | — | 0.0027 | 1.0000 |

## Conversational & Meeting Diarization SV Benchmarks

Evaluated on overlap-subtracted (`0.10 s` collar) speaker-disjoint test splits across Fisher (8 kHz -> 16 kHz telephone), CallHome English (telephone), AMI (`Mix-Headset` 4-speaker meetings), and ICSI (multi-party meetings):

| Benchmark Suite | Trials | Raw EER (%) | AS-Norm EER (%) | AS-Norm + QMF EER (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Fisher Test — Within-Session** | 10,000 | **1.82%** | **1.80%** | **1.38%** |
| **Fisher Test — Standard** | 10,000 | **1.31%** | **1.42%** | **0.97%** |
| **Fisher Test — Short-Turn (0.8-2.0s)** | 8,000 | **2.19%** | **2.23%** | **1.88%** |
| **AMI Test — Within-Session** | 8,000 | **6.81%** | **7.03%** | **5.51%** |
| **AMI Test — Standard** | 8,000 | **6.60%** | **6.41%** | **5.75%** |
| **AMI Test — Short-Turn (0.8-2.0s)** | 6,000 | **8.38%** | **8.08%** | **7.75%** |
| **ICSI Test — Within-Session** | 6,000 | **10.38%** | **9.82%** | **9.08%** |
| **ICSI Test — Standard** | 6,000 | **15.58%** | **14.82%** | **12.87%** |
| **ICSI Test — Short-Turn (0.8-2.0s)** | 4,000 | **15.88%** | **15.05%** | **14.57%** |
| **CallHome Test — Within-Session** | 6,000 | **3.38%** | **3.70%** | **3.17%** |
| **CallHome Test — Standard** | 6,000 | **3.20%** | **3.67%** | **3.08%** |
| **CallHome Test — Short-Turn (0.8-2.0s)** | 4,000 | **3.27%** | **3.65%** | **3.50%** |
| **4-Corpus Macro-Avg — Within-Session** | — | **5.60%** | **5.59%** | **4.79%** |
| **4-Corpus Macro-Avg — Standard** | — | **6.67%** | **6.58%** | **5.67%** |
| **4-Corpus Macro-Avg — Short-Turn** | — | **7.43%** | **7.25%** | **6.92%** |

## Multi-Domain Evaluation Results

### 1-Utterance Verification Across Evaluation Domains

Trial lists are 50/50 target/impostor over held-out speakers (LibriSpeech 40, CN-Celeb2 30, VoxCeleb1 10 speakers); minDCF@0.01 is close to its trivial ceiling of 1.0 for every model under this trial mix.

| Evaluation Split | Trials | Raw EER (%) | AS-Norm EER (%) | 95% Bootstrap CI (%) | minDCF@0.01 | minDCF@0.10 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LibriSpeech (English Read)** | 3,000 | **0.80%** | **0.83%** | [0.43, 1.15] | 0.0167 | 0.0167 | 0.9998 |
| **CN-Celeb2 (Chinese Multi-Genre)** | 3,000 | **7.83%** | **8.37%** | [6.96, 8.87] | 0.2787 | 0.2127 | 0.9778 |
| **VoxCeleb1 (In-the-Wild Celebrity)** | 2,000 | **0.50%** | **0.40%** | [0.20, 0.85] | 0.0170 | 0.0170 | 0.9999 |
| **Universal Pooled (All 3 Domains)** | 8,000 | **4.04%** | **4.51%** | [3.56, 4.50] | 0.1340 | 0.1053 | 0.9922 |
| **Universal Macro-Average (3 Domains)** | — | **3.04%** | **3.20%** | — | — | — | — |

### 3-Utterance Multi-Enrollment Verification Across Evaluation Domains

| Evaluation Split | Trials | 3-Utt EER (%) | 95% Bootstrap CI (%) | minDCF@0.01 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LibriSpeech (3-Utt)** | 1,000 | **0.50%** | [0.00, 0.90] | 0.0040 | 0.9998 |
| **CN-Celeb2 (3-Utt)** | 1,000 | **4.20%** | [2.95, 5.60] | 0.1300 | 0.9946 |
| **VoxCeleb1 (3-Utt)** | 1,000 | **0.00%** | [0.00, 0.00] | 0.0000 | 1.0000 |
| **Universal Pooled (3-Utt)** | 3,000 | **1.90%** | [1.40, 2.60] | 0.0453 | 0.9985 |
| **Universal Macro-Average (3-Utt)** | — | **1.57%** | — | — | — |

Single seed; the 95% bootstrap CIs resample trials (not speakers). Differences below about 1% absolute EER between FlaxSpeaker checkpoints are within the seed-to-seed variation measured in `pretrained_models/r2_seed_and_control_results.json` (see the GitHub README) and should not be interpreted as rankings.

## Deviations from the Source Papers

- **Training regime**: 78,000 gradient steps x 512 utterances (~15.06 epochs) across 24,110 base speakers (2,481,542 clean utterances + 3x speed perturbation + MUSAN/RIR).
- **Augmentation**: corpus-backed MUSAN environmental noise, music, and multi-speaker babble + RIRS_NOISES room impulse responses + 3-way polyphase speed perturbation (`0.9x`, `1.0x`, `1.1x`) + SpecAugment.
- **SimAM-ERes2Conformer**: unified 2D multi-scale `SimAM-SERes2Net` (`base_channels=64`, `num_blocks=(3, 4, 6, 3)`, `scale=4`, `width_mult=2`, `SimAM+2D-fwSE+AFF`, `temporal_strides=(1, 1, 2, 2)`) + `3-Stage (2+3+4) MFA` + 3x `512`-D `RoPE+QK-Norm+BatchNorm` Conformer blocks + `catp` + `BN-Neck`.
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

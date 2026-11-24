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
- conformer
- extended_set_softmax
- pfas
datasets:
- librispeech_asr
- cnceleb
- voxceleb
metrics:
- eer
- min_dcf
- roc_auc
---

# FlaxSpeaker: `univ_full_conformer_large_pfas` (`conformer` - `large`)

## Disclaimer & Reproduction Notice

> **This is not an officially supported Google product.**
>
> This checkpoint and the accompanying `FlaxSpeaker` library are a from-scratch re-implementation in JAX/Flax, written using only the published papers (no proprietary code or data), of speaker recognition algorithms from:
> - **GE2E**: *Generalized End-to-End Loss for Speaker Verification* ([arXiv:1710.10467](https://arxiv.org/abs/1710.10467))
> - **Dr-Vectors**: *Decision Residual Networks and an Improved Loss for Speaker Recognition* ([arXiv:2104.01989](https://arxiv.org/abs/2104.01989))
> - **ATP** (pooling head repurposed from a *language identification* paper): *Attentive Temporal Pooling for Conformer-based Streaming Language Identification in Long-form Speech* ([arXiv:2202.12163](https://arxiv.org/abs/2202.12163))
> - **PFAS**: *Parameter-Free Attentive Scoring for Speaker Verification* ([arXiv:2203.05642](https://arxiv.org/abs/2203.05642))
>
> The original papers were run on Google-internal infrastructure and data; this checkpoint is a **full-data open-corpus universal flagship** trained on **12,193 speakers (464,557 clean utterances + MUSAN/RIR/speed augmentation)** across VoxCeleb2-dev, VoxCeleb1-dev, CN-Celeb 1+2, and LibriSpeech 960h with zero speaker overlap against held-out evaluation sets.

## Training Budget (read this first)

This flagship model was trained for **24,000 gradient steps** (warmup `1000` steps, batch size `256`) over the full multi-corpus universal training pool (VoxCeleb1/2 + CN-Celeb1/2 + LibriSpeech 960h) (12,193 base speakers, 464,557 clean utterances, expanded via 3-way speed perturbation + MUSAN/RIR).

**Training-crop provenance.** This checkpoint was trained with `U(140, 200)`-frame variable-length crops and an exact validity mask (the current pipeline default); the shipped `config.json` / YAML (`variable_length_training: true`) describes the released weights. Step-0 loss fingerprint at the released seed (`scripts/fingerprint_released_checkpoints.py`, `pretrained_models/step0_fingerprints.json`): archived `2.4578`; current code with fixed 200-frame crops `2.3256`, with `U(140, 200)` crops `2.458` (match: `variable_approx`). The variable-crop CPU `float32` probe is within `0.0002` of the archived TPU `bfloat16` step-0 loss, whereas the fixed-crop arm is more than 5x farther away. Evaluating the released weights with the current code reproduces the archived metrics in `eval_results.json`. At the converged `S_safe` budget, the seed-to-seed standard deviation of pooled 1-utt EER across seeds 0-2 is 0.08-0.41 % absolute (`pretrained_models/r2_seed_and_control_results.json`), while the 8,000-trial 95 % bootstrap CI half-width is ~0.8 %.

## Model Summary

| Property | Value |
| :--- | :--- |
| **Model ID** | `univ_full_conformer_large_pfas` |
| **Hugging Face Repository** | [`wq2012/univ_full_conformer_large_pfas`](https://huggingface.co/wq2012/univ_full_conformer_large_pfas) |
| **Hugging Face Collection** | [FlaxSpeaker Collection](https://huggingface.co/collections/wq2012/flaxspeaker) |
| **GitHub Repository** | [`wq2012/FlaxSpeaker`](https://github.com/wq2012/FlaxSpeaker) |
| **PyPI Package** | [`flaxspeaker`](https://pypi.org/project/flaxspeaker/) |
| **Backbone Architecture** | `conformer` (`large`) |
| **Temporal Pooling** | `pfas` |
| **Training Objective** | Blended Extended-Set Softmax (FlaxSpeaker variant: 0.50 row-softmax + 0.50 extended-set with a -log N logit shift; not the paper's Eq. 2) (`extended_set_softmax`) |
| **Verification Scorer** | PFAS-SN (FlaxSpeaker extension of PFAS: `shared_nonlinear` frame attention + global ATP anchor; 8 keys x (32 + 128); paper scoring Eqs. 5-7) (`pfas`) |
| **Output Representation Dim** | `1280` |
| **Inference Parameters (encoder + scoring head; excludes training-only classifier weights)** | `22,582,159` |
| **Training Data** | VoxCeleb2-dev + VoxCeleb1-dev + CN-Celeb1/2 + LibriSpeech 960h (full multi-domain SOTA flagship, 12,193 speakers, 464,557 utterances) |
| **Training Steps / Batch** | `24,000` steps x 256 utterances (~13.23 epochs), seed 42 |
| **Audio Frontend** | `80`-dim natural-log Mel @ `16000` Hz + frame-level energy VAD + CMVN (`flaxspeaker.AudioFrontend`; **not** a Hugging Face `AutoFeatureExtractor`) |

## Model Files Included

- `model.safetensors`: model weights (encoder + GE2E `w,b` / Dr-Vector / PFAS heads) in SafeTensors format
- `flax_model.msgpack`: native Flax Msgpack serialized parameters
- `config.json`: `FlaxSpeakerConfig` (full `ExperimentConfig`, incl. the frontend/VAD settings)
- `eval_results.json`: multi-domain evaluation metrics (1-utt & 3-utt EER, minDCF, ROC-AUC, 95% bootstrap CIs, VoxCeleb1-O 10-speaker subset)

## Quickstart Usage

### Python API (from Hugging Face Hub or Local Directory)

Requires `flaxspeaker >= 0.8.0` (`pip install -U flaxspeaker`).

```python
from flaxspeaker import load_pretrained

model = load_pretrained("wq2012/univ_full_conformer_large_pfas")
emb = model.extract_embedding("path/to/utterance.flac")
score = model.verify("path/to/enroll.flac", "path/to/test.flac")
print(f"Embedding shape: {emb.shape}, Verification score: {score:.4f}")
```

### Command-Line Inference

```bash
python scripts/inference.py --model_dir wq2012/univ_full_conformer_large_pfas \
    --enroll_audio path/to/enroll.flac --test_audio path/to/test.flac
```

## Official Zero-Shot Speaker Verification Benchmarks

| Benchmark | Trials | Raw EER (%) | AS-Norm EER (%) | AS-Norm + QMF EER (%) | minDCF@0.01 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **VoxCeleb1-O (`veri_test2.txt`, 40 spks)** | 37,611 | **2.79%** | **2.78%** | **2.57%** | 0.3134 | 0.9965 |
| **CN-Celeb-E (`eval/lists/trials.lst`, 200 spks)** | 3,484,292 | **12.36%** | **11.94%** | **11.18%** | 0.6272 | 0.9472 |
| **CN-Celeb-E Multi-Enroll (`enroll.map`)** | 3,484,292 | **11.02%** | **10.43%** | — | 0.6421 | 0.9573 |
| **LibriSpeech `test-clean` (40 spks)** | 10,000 | **2.05%** | **2.12%** | **1.99%** | 0.1382 | 0.9971 |
| **LibriSpeech `test-clean` (3-Utt)** | 3,000 | **0.97%** | **0.77%** | — | 0.0493 | 0.9993 |

## Multi-Domain Evaluation Results

### 1-Utterance Verification Across Evaluation Domains

Trial lists are 50/50 target/impostor over held-out speakers (LibriSpeech 40, CN-Celeb2 30, VoxCeleb1 10 speakers); minDCF@0.01 is close to its trivial ceiling of 1.0 for every model under this trial mix.

| Evaluation Split | Trials | Raw EER (%) | AS-Norm EER (%) | 95% Bootstrap CI (%) | minDCF@0.01 | minDCF@0.10 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LibriSpeech (English Read)** | 3,000 | **2.37%** | **2.27%** | [1.78, 2.90] | 0.0653 | 0.0653 | 0.9974 |
| **CN-Celeb2 (Chinese Multi-Genre)** | 3,000 | **11.73%** | **12.60%** | [10.60, 12.92] | 0.5220 | 0.3813 | 0.9544 |
| **VoxCeleb1 (In-the-Wild Celebrity)** | 2,000 | **3.05%** | **2.70%** | [2.20, 3.80] | 0.1650 | 0.1150 | 0.9966 |
| **Universal Pooled (All 3 Domains)** | 8,000 | **6.36%** | **6.79%** | [5.83, 6.87] | 0.2822 | 0.2187 | 0.9834 |
| **Universal Macro-Average (3 Domains)** | — | **5.72%** | **5.86%** | — | 0.2508 | 0.1872 | 0.9828 |

### 3-Utterance Multi-Enrollment Verification Across Evaluation Domains

| Evaluation Split | Trials | 3-Utt EER (%) | 95% Bootstrap CI (%) | minDCF@0.01 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LibriSpeech (3-Utt)** | 1,000 | **1.20%** | [0.40, 2.00] | 0.0320 | 0.9986 |
| **CN-Celeb2 (3-Utt)** | 1,000 | **7.60%** | [5.85, 9.00] | 0.3660 | 0.9807 |
| **VoxCeleb1 (3-Utt)** | 1,000 | **1.60%** | [0.60, 2.40] | 0.0260 | 0.9994 |
| **Universal Pooled (3-Utt)** | 3,000 | **3.87%** | [3.27, 4.62] | 0.2033 | 0.9940 |
| **Universal Macro-Average (3-Utt)** | — | **3.47%** | — | 0.1413 | 0.9929 |

Single seed; the 95% bootstrap CIs resample trials (not speakers). Differences below about 1% absolute EER between FlaxSpeaker checkpoints are within the seed-to-seed variation measured in `pretrained_models/r2_seed_and_control_results.json` (see the GitHub README) and should not be interpreted as rankings.

## Deviations from the Source Papers

- **Training regime**: 24,000 gradient steps x 256 utterances (~13.23 epochs) across 12,193 base speakers (464,557 clean utterances + 3x speed perturbation + MUSAN/RIR).
- **Augmentation**: corpus-backed MUSAN environmental noise, music, and multi-speaker babble + RIRS_NOISES room impulse responses + 3-way polyphase speed perturbation (`0.9x`, `1.0x`, `1.1x`) + SpecAugment.
- **Objective**: 50/50 blend of row-softmax and `-log N`-shifted extended-set softmax; the paper's pure Eq. 2 is available via `pure_extended_set: true`.
- **PFAS head**: `shared_nonlinear` multi-head tanh frame attention with a global ATP anchor and 8 keys x (32 + 128) (FlaxSpeaker 'PFAS-SN'); the paper uses ATP -> affine+ReLU -> linear with 32 tied keys (`odyssey2022_pfas_config(pure_paper_mode=True)`).
- **Conformer**: parallel linear + Conv1D stem, no positional encoding, 6 blocks x 384-D, 8 heads, kernel 15, mask-aware GroupNorm(2), no dropout, multi-layer feature aggregation; the Google papers use 12 x 512 blocks, 4-8 heads, kernel 32, relative positional encoding and dropout.
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

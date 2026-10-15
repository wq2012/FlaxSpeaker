---
license: apache-2.0
library_name: flaxspeaker
pipeline_tag: audio-classification
tags:
- audio
- speaker-verification
- speaker-recognition
- speaker-embeddings
- jax
- flax
- safetensors
- conformer
- arcface
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

# FlaxSpeaker: `ls_conformer_small_arcface` (`conformer` - `small`)

## Disclaimer & Open-Source Reproduction Notice

> **This is not an officially supported Google product.**
>
> This model checkpoint and the accompanying `FlaxSpeaker` library are an independent, clean-room open-source reproduction in JAX/Flax of speaker recognition algorithms published in:
> - **GE2E**: *Generalized End-to-End Loss for Speaker Verification* ([arXiv:1710.10467](https://arxiv.org/abs/1710.10467))
> - **Dr-Vectors**: *Decision Residual Networks and an Improved Loss for Speaker Recognition* ([arXiv:2104.01989](https://arxiv.org/abs/2104.01989))
> - **Conformer-ATP**: *Attentive Temporal Pooling for Conformer-based Streaming Language Identification in Long-form Speech* ([arXiv:2202.12163](https://arxiv.org/abs/2202.12163))
> - **PFAS**: *Parameter-Free Attentive Scoring for Speaker Verification* ([arXiv:2203.05642v3](https://arxiv.org/abs/2203.05642v3))
>
> The experiments in those original papers were conducted using Google's proprietary internal infrastructure and internal datasets, not this open-source library or public dataset split.

## Model Summary

| Property | Value |
| :--- | :--- |
| **Model ID** | `ls_conformer_small_arcface` |
| **Hugging Face Repository** | [`wq2012/ls_conformer_small_arcface`](https://huggingface.co/wq2012/ls_conformer_small_arcface) |
| **Hugging Face Collection** | [FlaxSpeaker Collection](https://huggingface.co/collections/wq2012/flaxspeaker-6ad02d999a4180c794870595) |
| **GitHub Repository** | [`wq2012/FlaxSpeaker`](https://github.com/wq2012/FlaxSpeaker) |
| **PyPI Package** | [`flaxspeaker`](https://pypi.org/project/flaxspeaker/) |
| **Backbone Architecture** | `conformer` (`small`) |
| **Temporal Pooling** | `atp` |
| **Training Loss** | `arcface` |
| **Verification Scoring** | `cosine` |
| **Output Embedding Dim** | `192` |
| **Encoder Parameters** | `2,029,892` |
| **Training Domain** | `librispeech` |
| **Audio Frontend** | `80`-dim `log_mel` @ `16000` Hz + Energy VAD + CMVN |

## Model Files Included

- `model.safetensors`: Primary model weights in SafeTensors format
- `flax_model.msgpack`: Native Flax Msgpack serialized parameters
- `config.json`: `FlaxSpeakerConfig` model & training architecture configuration
- `preprocessor_config.json`: `WhisperFeatureExtractor`-compatible frontend configuration
- `eval_results.json`: Verified multi-domain evaluation metrics (1-utt & 3-utt EER, minDCF, ROC-AUC, 95% bootstrap CIs)

## Quickstart Usage

### Python API (from Hugging Face Hub or Local Directory)

```python
from flaxspeaker import load_pretrained

model = load_pretrained("wq2012/ls_conformer_small_arcface")
emb = model.extract_embedding("path/to/utterance.flac")
score = model.verify("path/to/enroll.flac", "path/to/test.flac")
print(f"Embedding shape: {emb.shape}, Verification score: {score:.4f}")
```

### Command-Line Inference

```bash
python scripts/inference.py --model_dir wq2012/ls_conformer_small_arcface \
    --enroll_audio path/to/enroll.flac --test_audio path/to/test.flac
```

## Multi-Domain Evaluation Results

### 1-Utterance Verification Across Evaluation Domains

| Evaluation Split | Trials | EER (%) | 95% Bootstrap CI (%) | minDCF@0.01 | minDCF@0.10 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LibriSpeech (English Read)** | 3,000 | **14.47%** | [13.20, 15.72] | 0.9433 | 0.6947 | 0.9312 |
| **CN-Celeb2 (Chinese Multi-Genre)** | 3,000 | **28.73%** | [27.03, 30.40] | 0.9833 | 0.9200 | 0.7829 |
| **VoxCeleb1 (In-the-Wild Celebrity)** | 2,000 | **23.85%** | [22.05, 25.90] | 0.9620 | 0.9320 | 0.8491 |
| **Universal Pooled (All 3 Domains)** | 8,000 | **22.80%** | [21.91, 23.69] | 0.9820 | 0.8630 | 0.8581 |
| **Universal Macro-Average (3 Domains)** | 8,000 | **22.35%** | — | 0.9629 | 0.8489 | 0.8544 |

### 3-Utterance Multi-Enrollment Verification Across Evaluation Domains

| Evaluation Split | Trials | 3-Utt EER (%) | 95% Bootstrap CI (%) | minDCF@0.01 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LibriSpeech (3-Utt)** | 1,000 | **9.60%** | [7.60, 11.90] | 0.8180 | 0.9657 |
| **CN-Celeb2 (3-Utt)** | 1,000 | **25.50%** | [23.05, 28.20] | 0.9860 | 0.8300 |
| **VoxCeleb1 (3-Utt)** | 1,000 | **18.30%** | [15.55, 20.60] | 0.9520 | 0.9017 |
| **Universal Pooled (3-Utt)** | 3,000 | **18.23%** | [16.83, 19.50] | 0.9713 | 0.9056 |
| **Universal Macro-Average (3-Utt)** | 3,000 | **17.80%** | — | 0.9187 | 0.8991 |

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

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

# FlaxSpeaker: `univ_conformer_small_pfas` (`conformer` - `small`)

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
| **Model ID** | `univ_conformer_small_pfas` |
| **Hugging Face Repository** | [`wq2012/univ_conformer_small_pfas`](https://huggingface.co/wq2012/univ_conformer_small_pfas) |
| **Hugging Face Collection** | [FlaxSpeaker Collection](https://huggingface.co/collections/wq2012/flaxspeaker-6ad02d999a4180c794870595) |
| **GitHub Repository** | [`wq2012/FlaxSpeaker`](https://github.com/wq2012/FlaxSpeaker) |
| **PyPI Package** | [`flaxspeaker`](https://pypi.org/project/flaxspeaker/) |
| **Backbone Architecture** | `conformer` (`small`) |
| **Temporal Pooling** | `pfas` |
| **Training Loss** | `extended_set_softmax` |
| **Verification Scoring** | `pfas` |
| **Output Embedding Dim** | `640` |
| **Encoder Parameters** | `2,162,187` |
| **Training Domain** | `universal` |
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

model = load_pretrained("wq2012/univ_conformer_small_pfas")
emb = model.extract_embedding("path/to/utterance.flac")
score = model.verify("path/to/enroll.flac", "path/to/test.flac")
print(f"Embedding shape: {emb.shape}, Verification score: {score:.4f}")
```

### Command-Line Inference

```bash
python scripts/inference.py --model_dir wq2012/univ_conformer_small_pfas \
    --enroll_audio path/to/enroll.flac --test_audio path/to/test.flac
```

## Multi-Domain Evaluation Results

### 1-Utterance Verification Across Evaluation Domains

| Evaluation Split | Trials | EER (%) | 95% Bootstrap CI (%) | minDCF@0.01 | minDCF@0.10 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LibriSpeech (English Read)** | 3,000 | **15.83%** | [14.53, 17.22] | 0.8247 | 0.7313 | 0.9252 |
| **CN-Celeb2 (Chinese Multi-Genre)** | 3,000 | **23.17%** | [21.42, 24.47] | 0.9153 | 0.8207 | 0.8635 |
| **VoxCeleb1 (In-the-Wild Celebrity)** | 2,000 | **22.95%** | [20.90, 24.85] | 0.9500 | 0.9330 | 0.8637 |
| **Universal Pooled (All 3 Domains)** | 8,000 | **20.35%** | [19.48, 21.21] | 0.8972 | 0.8295 | 0.8888 |
| **Universal Macro-Average (3 Domains)** | 8,000 | **20.65%** | — | 0.8967 | 0.8283 | 0.8841 |

### 3-Utterance Multi-Enrollment Verification Across Evaluation Domains

| Evaluation Split | Trials | 3-Utt EER (%) | 95% Bootstrap CI (%) | minDCF@0.01 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LibriSpeech (3-Utt)** | 1,000 | **10.80%** | [8.80, 12.85] | 0.6600 | 0.9617 |
| **CN-Celeb2 (3-Utt)** | 1,000 | **18.60%** | [16.20, 21.00] | 0.9140 | 0.9008 |
| **VoxCeleb1 (3-Utt)** | 1,000 | **18.00%** | [15.90, 21.00] | 0.8440 | 0.9027 |
| **Universal Pooled (3-Utt)** | 3,000 | **16.00%** | [14.73, 17.33] | 0.9093 | 0.9251 |
| **Universal Macro-Average (3-Utt)** | 3,000 | **15.80%** | — | 0.8060 | 0.9217 |

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

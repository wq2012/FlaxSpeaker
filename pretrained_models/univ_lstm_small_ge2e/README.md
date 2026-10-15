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
- lstm
- ge2e_softmax
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

# FlaxSpeaker: `univ_lstm_small_ge2e` (`lstm` - `small`)

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
| **Model ID** | `univ_lstm_small_ge2e` |
| **Hugging Face Repository** | [`wq2012/univ_lstm_small_ge2e`](https://huggingface.co/wq2012/univ_lstm_small_ge2e) |
| **Hugging Face Collection** | [FlaxSpeaker Collection](https://huggingface.co/collections/wq2012/flaxspeaker-6ad02d999a4180c794870595) |
| **GitHub Repository** | [`wq2012/FlaxSpeaker`](https://github.com/wq2012/FlaxSpeaker) |
| **PyPI Package** | [`flaxspeaker`](https://pypi.org/project/flaxspeaker/) |
| **Backbone Architecture** | `lstm` (`small`) |
| **Temporal Pooling** | `asp` |
| **Training Loss** | `ge2e_softmax` |
| **Verification Scoring** | `cosine` |
| **Output Embedding Dim** | `192` |
| **Encoder Parameters** | `1,710,914` |
| **Training Domain** | `universal` |
| **Audio Frontend** | `80`-dim `log_mel` @ `16000` Hz + Energy VAD + CMVN |

## Model Files Included

- `model.safetensors`: Primary model weights in SafeTensors format
- `flax_model.msgpack`: Native Flax Msgpack serialized parameters
- `config.json`: `FlaxSpeakerConfig` model & training architecture configuration
- `preprocessor_config.json`: `WhisperFeatureExtractor`-compatible frontend configuration
- `eval_results.json`: Verified multi-domain evaluation metrics (1-utt & 3-utt EER, minDCF, ROC-AUC, 95% bootstrap CIs)
- `model_fp32.tflite`: Exported FP32 TensorFlow Lite model for edge/mobile CPU inference
- `model_int8.tflite`: Dynamic-range INT8 quantized TensorFlow Lite model (~3.5x smaller)

## Quickstart Usage

### Python API (from Hugging Face Hub or Local Directory)

```python
from flaxspeaker import load_pretrained

model = load_pretrained("wq2012/univ_lstm_small_ge2e")
emb = model.extract_embedding("path/to/utterance.flac")
score = model.verify("path/to/enroll.flac", "path/to/test.flac")
print(f"Embedding shape: {emb.shape}, Verification score: {score:.4f}")
```

### Command-Line Inference

```bash
python scripts/inference.py --model_dir wq2012/univ_lstm_small_ge2e \
    --enroll_audio path/to/enroll.flac --test_audio path/to/test.flac
```

## Multi-Domain Evaluation Results

### 1-Utterance Verification Across Evaluation Domains

| Evaluation Split | Trials | EER (%) | 95% Bootstrap CI (%) | minDCF@0.01 | minDCF@0.10 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LibriSpeech (English Read)** | 3,000 | **14.97%** | [13.80, 16.35] | 0.9493 | 0.7667 | 0.9236 |
| **CN-Celeb2 (Chinese Multi-Genre)** | 3,000 | **25.53%** | [23.83, 26.98] | 0.9787 | 0.9660 | 0.8305 |
| **VoxCeleb1 (In-the-Wild Celebrity)** | 2,000 | **22.50%** | [20.70, 24.20] | 0.9580 | 0.9070 | 0.8643 |
| **Universal Pooled (All 3 Domains)** | 8,000 | **21.31%** | [20.42, 22.15] | 0.9750 | 0.9028 | 0.8730 |
| **Universal Macro-Average (3 Domains)** | 8,000 | **21.00%** | — | 0.9620 | 0.8799 | 0.8728 |

### 3-Utterance Multi-Enrollment Verification Across Evaluation Domains

| Evaluation Split | Trials | 3-Utt EER (%) | 95% Bootstrap CI (%) | minDCF@0.01 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LibriSpeech (3-Utt)** | 1,000 | **11.20%** | [9.20, 13.10] | 0.7840 | 0.9630 |
| **CN-Celeb2 (3-Utt)** | 1,000 | **21.60%** | [18.75, 23.60] | 0.9860 | 0.8732 |
| **VoxCeleb1 (3-Utt)** | 1,000 | **19.50%** | [16.60, 21.80] | 0.9820 | 0.8904 |
| **Universal Pooled (3-Utt)** | 3,000 | **18.00%** | [16.67, 19.33] | 0.9853 | 0.9082 |
| **Universal Macro-Average (3-Utt)** | 3,000 | **17.43%** | — | 0.9173 | 0.9089 |

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

# Inference Guide

This document covers 1. an inference demo with RADAR pre-trained checkpoint on RAD-CT, and 2. using RADAR pre-trained checkpoint to inference and evaluate on the external MERLIN test set.

> **Prerequisite:** Complete the [Setup](../README.md#setup) section in the main README first.

---

## Table of Contents

- [Inference Demo](#1-inference-demo)
- [Inference on the external MERLIN Test Set](#(Optional)-2-inference-on-the-external-merlin-test-set)

---

## 1. Inference Demo

### (1) Download Required Files and Place Them to Target folder

| File                  | Description                            | Destination                                                        |
| --------------------- | -------------------------------------- | ------------------------------------------------------------------ |
| Pretrained checkpoint | RADAR pre-trained checkpoint on RAD-CT | `radar/RADAR_inference/checkpoint/checkpoint_radar_pretrain.pth` |
| `bert-base-chinese` | BERT tokenizer and model               | `radar/RADAR_inference/configs/bert-base-chinese/`               |

<!-- | `RADAR_infer_results_MerlinTestset.csv`   | The inference results of RADAR on Merlin-CT-Test set | `radar/RADAR_inference/RADAR_infer_results_MerlinTestset.csv`      | -->

<!-- | Demo cases                                | Sample CT examinations                               | `radar/RADAR_inference/demo_cases/`                                | -->

All files are available on HuggingFace.

### (2) Run Inference

```bash
cd RADAR_inference
python inference_demo.py
```

The results will be saved as `RADAR_infer_results_demo.csv`, which contains the positive scores for each finding.

---

## (Optional) 2. Inference on the External MERLIN Test Set

### (1) Prepare Data

- Download the MERLIN dataset from the [Stanford AIMI Shared Datasets](https://huggingface.co/datasets/stanfordaimi/merlin).
- Modify `--img_dir` to the MERLIN data path on your local device in `inference_merlin_testset.py`.
- We provide all MERLIN reports in JSON format (`merlin_report.json`), which includes the test split information.
- The labels have also been converted to JSON format for convenience (`merlin_labels.json`).

### (2) Run Inference

Single-GPU:

```bash
cd RADAR_inference
python inference_merlin_testset.py
```

Multi-GPU (recommended):

```bash
cd RADAR_inference
torchrun --nproc_per_node=8 inference_merlin_testset.py
```

The inference results will be saved as a CSV file. (We also provided for convenience: `RADAR_infer_results_MerlinTestset.csv`)

### (3) Evaluate Performance

Compute performance metrics using the inference CSV and the label file:

```bash
python calc_metrics_merlin_testset.py
```

---

[← Back to README](../README.md)

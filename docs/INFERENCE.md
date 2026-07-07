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

| File                  | Description                            | Destination                            |
| --------------------- | -------------------------------------- | -------------------------------------- |
| Pretrained checkpoint | RADAR pre-trained checkpoint on RAD-CT | `ckpt/checkpoint_radar_pretrain.pth` |
| `bert-base-chinese` | BERT tokenizer and model               | `ckpt/bert-base-chinese`             |

<!-- | `RADAR_infer_results_MerlinTestset.csv`   | The inference results of RADAR on Merlin-CT-Test set | `radar/RADAR_inference/RADAR_infer_results_MerlinTestset.csv`      | -->

<!-- | Demo cases                                | Sample CT examinations                               | `radar/RADAR_inference/demo_cases/`                                | -->

All files are available on HuggingFace.

### (2) Run Inference

```bash
cd RADAR_inference
python inference_demo.py
```

We have provided a demo nifty, the results will be saved as `RADAR_infer_results_demo.csv`, which contains the positive scores for each finding.

---

## (Optional) 2. Inference on the External MERLIN Test Set

### (1) Prepare Data

- Download the MERLIN dataset from the [Stanford AIMI Shared Datasets](https://stanfordaimi.azurewebsites.net/datasets/60b9c7ff-877b-48ce-96c3-0194c8205c40).
- Modify `--img_dir` to the MERLIN data path on your local device in `inference_merlin_testset.py`.
- We provide all MERLIN reports in JSON format (`merlin_report.json`), which includes the test split information.
- The labels have also been converted to JSON format for convenience (`merlin_labels.json`).

### (2) Run Inference

```bash
cd RADAR_inference
python inference_merlin_testset.py
```

The inference results will be saved as a CSV file. (We also provided for convenience: `RADAR_infer_results_MerlinTestset.csv`)

### (3) Evaluate Performance

Compute performance metrics using the inference CSV and the label file:

```bash
python calc_metrics_merlin_testset.py
```

The script reports the AUC for each finding, as well as the average AUC across all findings:

```bash
abdominal_aortic_aneurysm 0.9903
atherosclerosis 0.8739
submucosal_edema 0.8879
appendicitis 0.7621
bowel_obstruction 0.9704
aortic_valve_calcification 0.8436
cardiomegaly 0.8724
biliary_ductal_dilation 0.8711
hepatomegaly 0.8988
hepatic_steatosis 0.8917
pleural_effusion 0.9574
atelectasis 0.7091
renal_hypodensities 0.9122
renal_cyst 0.9426
hydronephrosis 0.88
gallstones 0.9193
pancreatic_atrophy 0.9356
splenomegaly 0.9682
fracture 0.6834
hiatal_hernia 0.8601
surgically_absent_gallbladder 0.9234
AvgAUC: 0.8835
```

The table below summarizes the evaluation performance.

| model | Description       | AUC           | ckpt path                              |
| ----- | ----------------- | ------------- | -------------------------------------- |
| RADAR | Trained on RAD-CT | 0.883         | `ckpt/checkpoint_radar_pretrain.pth`   |

---

[← Back to README](../README.md)

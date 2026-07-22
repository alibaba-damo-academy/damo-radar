# RADAR: An Expert-Level Generalist AI for Abdominal CT Diagnosis

[![GitHub](https://img.shields.io/badge/GitHub-Code-blue?logo=github&logoColor=white)](https://github.com/alibaba-damo-academy/damo-radar)
[![Zenodo](https://img.shields.io/badge/Zenodo-Code-7B61FF?logo=zenodo&logoColor=white)](https://zenodo.org/records/21271172)
[![Hugging Face](<https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model&Data-FFBF00>)](https://huggingface.co/radar-generalist)

RADAR is a generalist vision-language model trained on over 400,000 contrast-enhanced abdominal CT examinations with 15 million anatomy-aware image–text pairs, learning directly from clinical reports without manual annotation. RADAR provides a scalable and versatile framework for radiology AI, demonstrating expert-level performance across both routine and complex clinical tasks.

<p align="center">
  <img src="docs/radar_fig0.png" alt="RADAR Overview" width="90%">
</p>

---

## Setup

Create a conda environment and install the required dependencies:

```bash
conda create -n radar python=3.10
conda activate radar
pip install -r requirements.txt
```

> **Note:** The pinned package versions (e.g. `transformers==4.25`) follow the [LAVIS](https://github.com/salesforce/LAVIS). Other versions may also work.

---

## HuggingFace

- The pre-trained checkpoints and supporting files are available on [HuggingFace](https://huggingface.co/radar-generalist).
- For convenience, we have provided the demo nifty, and predicted results in CSV format in this repo. The supporting files required for the inference demo and training can be downloaded from HuggingFace.

---

## Zenodo

Code can also be archived on [Zenodo](https://zenodo.org/records/21271172).

---

## Documentation

For detailed instructions, please refer to the following guides:

| Guide                         | Description                                                                                                                                               |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Training](docs/TRAINING.md)   | Train RADAR/RADAR+ from scratch or fine-tune on MERLIN data; inference and evaluation are also included.                                                  |
| [Inference](docs/INFERENCE.md) | 1. An inference demo with a radar pre-trained checkpoint on RAD-CT, and 2. Inference and evaluation of radar performance on the external MERLIN test set. |
| [Preprocess](docs/PREPROCESS.md) | Image/mask and radiology report preprocessing code, which can be used to process the MERLIN data or your own custom data. |

---

## Acknowledgements

This project is built upon the following open-source projects:

- [LAVIS](https://github.com/salesforce/LAVIS)
- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)

---

## Citation

If you find RADAR useful in your research, please cite our paper:

```bibtex
@article{radar,
  title   = {RADAR: An Expert-Level Generalist AI for Abdominal CT Diagnosis},
  year    = {2026},
  note    = {BibTeX entry to be updated upon publication}
}
```

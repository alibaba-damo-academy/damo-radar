# RADAR: An Expert-Level Generalist AI for Abdominal CT Diagnosis

Artificial intelligence (AI) in radiology aspires to deliver expert-level diagnosis across diverse clinical tasks, yet existing supervised strategies remain limited in scope. We developed **RADAR**, a generalist vision–language model trained on more than **400,000** contrast-enhanced abdominal CT examinations and **15 million** anatomy-wise image–text pairs, learning directly from clinical reports without manual annotation. Across extensive internal and external evaluations spanning multiple centers and varied scenarios, RADAR achieved high diagnostic performance and robust generalization for **18 anatomical structures** and **146 imaging findings**. In a reader study, AI assistance increased the diagnostic sensitivity of 26 radiologists by approximately 10%, underscoring its clinical utility. In summary, RADAR offers a scalable, versatile, and interpretable solution, demonstrating that generalist AI can match human experts in both routine and complex radiology tasks.

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

---

## HuggingFace

- The pre-trained checkpoints and supports files are available on [HuggingFace](https://huggingface.co/radar-generalist).
- For convenience, we have provided the demo nifty, processed merlin-related reports and labels, predicted results in csv format in this repo. The supporting files required for inferecnce demo and training can be downloaded from HuggingFace.

---

## Documentation

For detailed instructions, please refer to the following guides:

| Guide                         | Description                                                                                                                               |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| [Inference](docs/INFERENCE.md) | 1. An inference demo with radar pre-trained checkpoint on RAD-CT, and 2. Inference and evaluate radar' performence on the MERLIN test set. |
| [Training](docs/TRAINING.md)   | Train, inference, and evaluate RADAR/RADAR+ from scratch or fine-tuning on MERLIN data.                                                   |

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

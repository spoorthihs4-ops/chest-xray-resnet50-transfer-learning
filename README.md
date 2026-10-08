# Chest X-Ray Classification with ResNet50 Transfer Learning

![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-CLAHE-5C3EE8?logo=opencv&logoColor=white)
![Kaggle](https://img.shields.io/badge/Trained%20on-Kaggle%20T4%20GPU-20BEFF?logo=kaggle&logoColor=white)

Classifies chest X-rays into **COVID-19, Normal, Pneumonia and Tuberculosis** by fine-tuning an ImageNet-pretrained **ResNet50** in three progressive phases, with CLAHE contrast enhancement, medically safe augmentation and class weighting for a heavily imbalanced dataset.

> MSc Data Science group project (Machine Learning), University of Hertfordshire. All results come from one complete run on a Kaggle Tesla T4 GPU.

---

## Highlights

| | |
|---|---|
| **Test accuracy** | **92.35 %** on 771 held-out X-rays (weighted F1 0.923) |
| **Per class** | COVID-19 and Tuberculosis **F1 = 1.00**; Pneumonia 0.93; Normal 0.87 |
| **Model** | ResNet50 (ImageNet) + custom 512 → 256 → 128 → 4 head, 24.8 M parameters |
| **Training** | 3-phase progressive unfreezing (head → top 55 layers → all 185 layers), cosine LR, label smoothing |
| **Data handling** | CLAHE preprocessing, balanced class weights, no horizontal flips (preserves heart position) |
| **Deployment** | Saved model + class map and a reusable `predict_xray()` function |

## Methodology

![Methodology flowchart](methodology_flowchart.png)

**[Explore the interactive methodology](https://spoorthihs4-ops.github.io/chest-xray-resnet50-transfer-learning/)** – click any stage to see what it does, why it matters, the result it produced and the techniques behind it.

<details>
<summary><b>Pipeline as a live diagram (zoom and pan on GitHub)</b></summary>

```mermaid
flowchart TB
    A["1 · Dataset & EDA<br/>7,135 X-rays · 4 classes"]:::c1 --> B["2 · Train / val / test<br/>5,378 / 948 / 771"]:::c2
    B --> C["3 · CLAHE preprocessing<br/>local contrast enhancement"]:::c3
    C --> D["4 · Safe augmentation<br/>no horizontal flips"]:::c4
    D --> E["5 · ResNet50 backbone<br/>ImageNet + custom head"]:::c5
    E --> F["6 · Imbalance & loss<br/>class weights · label smoothing"]:::c6
    F --> G["7 · 3-phase fine-tuning<br/>best val accuracy 98.4 %"]:::c7
    G --> H["8 · Evaluate on test<br/>accuracy 92.35 %"]:::c8
    H --> I["9 · TTA & error analysis<br/>Normal → Pneumonia errors"]:::c9
    I --> J["10 · Deploy & predict<br/>predict_xray()"]:::c10

    classDef c1 fill:#2B59C3,stroke:#1d3f8f,color:#fff
    classDef c2 fill:#0E9384,stroke:#0a6b60,color:#fff
    classDef c3 fill:#7A4FD1,stroke:#5a37a3,color:#fff
    classDef c4 fill:#B5338A,stroke:#862566,color:#fff
    classDef c5 fill:#E25A1C,stroke:#a8420f,color:#fff
    classDef c6 fill:#D99A06,stroke:#a37304,color:#fff
    classDef c7 fill:#1D7A8C,stroke:#135563,color:#fff
    classDef c8 fill:#D1335B,stroke:#9c2443,color:#fff
    classDef c9 fill:#16A34A,stroke:#0f7a37,color:#fff
    classDef c10 fill:#1F2A44,stroke:#0f1626,color:#fff
```
</details>

## Results

### Test set (771 held-out images)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| COVID-19 | 1.00 | 0.99 | 1.00 | 106 |
| Tuberculosis | 1.00 | 1.00 | 1.00 | 41 |
| Pneumonia | 0.90 | 0.95 | 0.93 | 390 |
| Normal | 0.91 | 0.83 | 0.87 | 234 |
| **Weighted average** | **0.924** | **0.924** | **0.923** | 771 |

| Phase | Layers trained | Learning rate | Best validation accuracy |
|---|---|---|---|
| 1 | Head only (backbone frozen) | 1e-3 | 96.31 % |
| 2 | Top 55 ResNet50 layers + head | 1e-4 | 97.78 % |
| 3 | All 185 layers | 5e-6 | **98.42 %** |

**Key findings**
- **Normal vs Pneumonia is the hard boundary.** About 17 % of Normal X-rays are predicted as Pneumonia, a known challenge because early pneumonia can look like a healthy lung.
- **Test-time augmentation hurt.** 10-pass TTA scored 84.70 % because Normal recall fell from 0.83 to 0.50, so the standard prediction is reported.
- **Validation is optimistic.** Validation images come from the training folder, so the 98.4 % validation score overstates performance; the 92.35 % held-out test score is the realistic estimate.

<p align="center">
  <img src="training_history.png" width="98%" alt="Training history across three phases">
</p>
<p align="center">
  <img src="confusion_matrices.png" width="60%" alt="Confusion matrices">
  <img src="val_test_tta_comparison.png" width="38%" alt="Validation vs test vs TTA">
</p>

## Notebooks

| Notebook | Contents |
|---|---|
| [`00_full_pipeline_run_all`](00_full_pipeline_run_all.ipynb) | **Everything, top to bottom - run this to reproduce** |
| [`01_dataset_exploration`](01_dataset_exploration.ipynb) | Dataset download, structure, summary statistics, image sizes, class distribution, sample X-rays |
| [`02_transfer_learning_model`](02_transfer_learning_model.ipynb) | Why ResNet50, configuration, CLAHE and augmentation, class weights, model architecture |
| [`03_three_phase_fine_tuning`](03_three_phase_fine_tuning.ipynb) | Cosine LR schedule, three training phases, combined training curves |
| [`04_evaluation_and_tta`](04_evaluation_and_tta.ipynb) | Validation and test metrics, TTA, confusion matrices, per-class and comparison charts |
| [`05_deployment_and_summary`](05_deployment_and_summary.ipynb) | Save and reload the model, `predict_xray()`, prediction grid, summary and reflection |

## Reproduce

1. Open [`00_full_pipeline_run_all.ipynb`](00_full_pipeline_run_all.ipynb) on **Kaggle** with a **GPU T4** accelerator and Internet enabled.
2. The dataset [`jtiptj/chest-xray-pneumoniacovid19tuberculosis`](https://www.kaggle.com/datasets/jtiptj/chest-xray-pneumoniacovid19tuberculosis) downloads automatically through `kagglehub`.
3. **Run All.** Each training epoch takes about 2.5-4 minutes on a T4 (longer once all layers are unfrozen).

## Limitations

- The validation subset is drawn from the training folder and also receives augmentation, so validation scores are optimistic and slightly noisy.
- Images may include several per patient; without patient IDs, a patient-level split cannot be guaranteed.
- This is a research prototype and is **not** a diagnostic tool.

## Contributors

Group project. Add team members and your own role here.

**Dr. Spoorthi H S** · MSc Data Science, University of Hertfordshire  
[LinkedIn](https://linkedin.com/in/dr-spoorthi-2005b6351) · [GitHub](https://github.com/spoorthihs4-ops)

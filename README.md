# ASL Sign Detector

American Sign Language gesture recognition using two approaches: a rule-based MediaPipe
classifier and fine-tuned EfficientNet models trained on 29 ASL classes. Deployed as a
Gradio app on Hugging Face Spaces.

**Live demo:** https://huggingface.co/spaces/d2j666/AAI_521_Final_Project

---

## Problem

Static ASL gesture recognition is a well-studied computer vision problem, but most
benchmarks use controlled studio images that don't reflect real webcam conditions.
We wanted to compare a geometry-based approach (no ML) against fine-tuned deep learning
models — and see how each held up outside the training distribution.

---

## Two Approaches

### 1. Rule-based (MediaPipe)

`model.py` uses MediaPipe Hands to extract 21 3D hand landmarks per frame, then classifies
gestures by analyzing finger extension states (tip position relative to PIP joint). No
training required. Covers 5 gestures: A, V, B, 1, W.

- Fast and runs locally with no model weights
- Brittle to hand orientation and partial occlusion
- Confidence is fixed per rule, not learned

### 2. ML-based (EfficientNet)

`model_ml.py` loads fine-tuned EfficientNet models (B4, B7, B9) from Hugging Face Hub
and runs image classification over 29 ASL classes (full alphabet + `del`, `space`, `nothing`).

Training used a 2-stage fine-tuning approach on the
[Kaggle ASL Alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
(87,000 images, 29 classes):

- **Stage 1**: Freeze EfficientNet backbone, train classification head (5 epochs, lr=1e-3)
- **Stage 2**: Unfreeze top layers, fine-tune end-to-end (5 epochs, lr=1e-4)

---

## Results

| Model | Parameters | Best Val Accuracy |
|---|---|---|
| EfficientNetB4 | 17.7M | 99.94% |
| EfficientNetB7 | 64.2M | 99.96% |
| EfficientNetB9 | 117.8M | 99.99% |

Validation accuracy was high across all three models — but real-world webcam performance
was significantly worse. The Kaggle ASL dataset uses clean studio backgrounds with
consistent lighting and standardized hand positions. Models trained on it overfit to
that distribution. On live webcam input with natural backgrounds and variable lighting,
classification broke down noticeably.

Larger models (B7, B9) did not meaningfully outperform B4 in practice, and added
substantial inference latency with no real-world benefit.

---

## Limitations

- **Distribution shift**: Near-perfect benchmark accuracy does not transfer to real webcam
  conditions. The training data is too clean relative to actual use.
- **No real-world test set**: We evaluated on a held-out split of the same Kaggle dataset,
  not on independently collected webcam footage. This masked the generalization gap.
- **Static gestures only**: The system classifies individual frames. Dynamic gestures
  (letters requiring motion, like J and Z) are not supported.
- **Rule-based coverage**: The MediaPipe classifier covers only 5 gestures. Extending it
  requires hand-crafting rules for each new gesture.

---

## How to Run

```bash
uv sync
uv run python app.py
```

App runs at `http://localhost:7860`. Three input modes: webcam snapshot, image upload,
and live streaming.

Models are downloaded automatically from Hugging Face Hub on first run.
Set `HF_TOKEN` in your environment if the model repo is private.

---

## Lessons Learned

The benchmark accuracy numbers looked strong, but the real test was webcam performance —
and that's where the gap showed. Evaluating only on a held-out split from the same dataset
gives a false sense of how well the model generalizes. A proper eval would include
independently collected samples under varied conditions.

The rule-based approach was more predictable in controlled settings despite its limited
gesture coverage. For production, a hybrid — MediaPipe for hand detection and cropping,
then a classifier on the cropped region — would likely outperform either approach alone.

---

## Team

Group project — AAI-521, University of San Diego.
Dennis Johnson (model training, app integration) · Wael · Ali

---

*Originally: `AAI_521_Final_Project` (AAI-521 — Computer Vision, University of San Diego)*

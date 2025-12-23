# Automated Bone Age Assessment (SVM + NN + Fusion)

This repo is **ready to run**. It follows the project pipeline in your documents:
- Preprocessing / normalization
- ROI extraction (whole hand, wrist, fingers/epiphysis)
- Feature extraction (HOG + intensity + histogram)
- Models: SVM (SVR-RBF) and Neural Network (MLP)
- Decision-level fusion
- Evaluation: MAE + tolerance accuracy (±12/±24 months)

---

## 1) Expected dataset structure (RSNA-style)

Put your dataset under a folder, e.g. `data/`:

```
data/
  train.csv   (columns: id, boneage, male [optional])
  test.csv    (columns: id, male [optional])
  train/      (PNG images named {id}.png)
  test/       (PNG images named {id}.png)
```

> Notes:
> - `boneage` is usually in **months** in Kaggle bone-age datasets.
> - If `male` is missing, the code automatically uses 0.

---

## 2) Install

```bash
python -m venv .venv
# Linux/Mac:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

pip install -r requirements.txt
```

---

## 3) Train (creates models + metrics)

```bash
python src/train.py --data_dir data --out_dir artifacts
```

Outputs in `artifacts/`:
- `scaler.joblib`, `svm.joblib`, `mlp.joblib`, `fusion_weights.joblib`
- `metrics.json`

---

## 4) Predict (creates submission)

```bash
python src/predict.py --data_dir data --artifacts_dir artifacts --out_csv submission.csv
```

---

## 5) Optional CNN (requires TensorFlow)

```bash
pip install tensorflow
python src/train.py --data_dir data --out_dir artifacts --use_cnn
```
# Automated Bone Age Assessment (SVM + NN + Fusion)

This repo is **ready to run**. It follows the project pipeline in your documents:
- Preprocessing / normalization
- ROI extraction (whole hand, wrist, fingers/epiphysis)
- Feature extraction (HOG + intensity + histogram)
- Models: SVM (SVR-RBF) and Neural Network (MLP)
- Decision-level fusion
- Evaluation: MAE + tolerance accuracy (±12/±24 months)

---

## 1) Expected dataset structure (RSNA-style)

Put your dataset under a folder, e.g. `data/`:

```
data/
  train.csv   (columns: id, boneage, male [optional])
  test.csv    (columns: id, male [optional])
  train/      (PNG images named {id}.png)
  test/       (PNG images named {id}.png)
```

> Notes:
> - `boneage` is usually in **months** in Kaggle bone-age datasets.
> - If `male` is missing, the code automatically uses 0.

---

## 2) Install

```bash
python -m venv .venv
# Linux/Mac:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

pip install -r requirements.txt
```

---

## 3) Train (creates models + metrics)

```bash
python src/train.py --data_dir data --out_dir artifacts
```

Outputs in `artifacts/`:
- `scaler.joblib`, `svm.joblib`, `mlp.joblib`, `fusion_weights.joblib`
- `metrics.json`

---

## 4) Predict (creates submission)

```bash
python src/predict.py --data_dir data --artifacts_dir artifacts --out_csv submission.csv
```

---

## 5) Optional CNN (requires TensorFlow)

```bash
pip install tensorflow
python src/train.py --data_dir data --out_dir artifacts --use_cnn
```

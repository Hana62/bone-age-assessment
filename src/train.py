import os
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from preprocessing import preprocess_image, extract_rois
from features import extract_features
from metrics import mae, tolerance_accuracy_months


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True,
                   help="Folder with train.csv and train/ images")
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--test_size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="artifacts")
    p.add_argument("--use_cnn", action="store_true",
                   help="Optional: train a small CNN regressor (requires tensorflow)")
    return p


def load_train_df(data_dir: str) -> pd.DataFrame:
    # Typical Kaggle RSNA format: train.csv has columns: id, boneage, male(optional)
    train_csv = os.path.join(data_dir, "train.csv")
    df = pd.read_csv(train_csv)

    required = {"id", "boneage"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"train.csv must contain columns {required}. Found: {list(df.columns)}")

    if "male" not in df.columns:
        df["male"] = 0

    return df


def make_feature_matrix(df: pd.DataFrame, data_dir: str, img_size: int):
    X_list = []
    y_list = []

    img_folder = os.path.join(data_dir, "train")
    if not os.path.isdir(img_folder):
        raise FileNotFoundError(f"Expected folder: {img_folder}")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        img_id = row["id"]
        path = os.path.join(img_folder, f"{img_id}.png")
        img = preprocess_image(path, out_size=(img_size, img_size))
        rois = extract_rois(img)
        feats = extract_features(rois)

        # add metadata feature(s)
        male = float(row.get("male", 0))
        feats = np.concatenate([feats, np.array([male], dtype=np.float32)], axis=0)

        X_list.append(feats)
        y_list.append(float(row["boneage"]))  # usually months in Kaggle RSNA

    X = np.vstack(X_list).astype(np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    return X, y


def train_classical_models(X_train, y_train):
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X_train)

    svm = SVR(kernel="rbf", C=50.0, gamma="scale", epsilon=3.0)
    svm.fit(Xs, y_train)

    mlp = MLPRegressor(
        hidden_layer_sizes=(512, 256),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=200,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=15,
        verbose=False,
    )
    mlp.fit(Xs, y_train)

    return scaler, svm, mlp


def predict_classical(scaler, svm, mlp, X):
    Xs = scaler.transform(X)
    p_svm = svm.predict(Xs)
    p_mlp = mlp.predict(Xs)
    return p_svm, p_mlp


def fit_fusion_weights(y_val, p1, p2):
    '''
    Decision-level fusion: learn linear weights on validation set
      y_hat = w1*p1 + w2*p2 + b
    '''
    A = np.vstack([p1, p2, np.ones_like(p1)]).T
    w, _, _, _ = np.linalg.lstsq(A, y_val, rcond=None)
    return w  # [w1, w2, b]


def apply_fusion(w, p1, p2):
    return w[0] * p1 + w[1] * p2 + w[2]


def train_optional_cnn(train_df, data_dir, img_size, out_dir):
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from sklearn.model_selection import train_test_split

    img_folder = os.path.join(data_dir, "train")

    X_imgs = []
    y = []
    male_feat = []
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Loading images for CNN"):
        img_id = row["id"]
        path = os.path.join(img_folder, f"{img_id}.png")
        img = preprocess_image(path, out_size=(img_size, img_size)).astype(np.float32) / 255.0
        img = np.stack([img, img, img], axis=-1)  # make 3-channel
        X_imgs.append(img)
        y.append(float(row["boneage"]))
        male_feat.append(float(row.get("male", 0)))

    X_imgs = np.asarray(X_imgs, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    male_feat = np.asarray(male_feat, dtype=np.float32).reshape(-1, 1)

    X_train_i, X_val_i, y_train_i, y_val_i, m_train, m_val = train_test_split(
        X_imgs, y, male_feat, test_size=0.15, random_state=42
    )

    base = tf.keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3)
    )
    base.trainable = False

    img_in = layers.Input(shape=(img_size, img_size, 3))
    meta_in = layers.Input(shape=(1,))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(img_in)
    x = base(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Concatenate()([x, meta_in])
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="linear")(x)

    model = models.Model([img_in, meta_in], out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mae")

    cb = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5),
    ]
    model.fit(
        [X_train_i, m_train], y_train_i,
        validation_data=([X_val_i, m_val], y_val_i),
        epochs=30, batch_size=16, callbacks=cb, verbose=1
    )

    model_path = os.path.join(out_dir, "cnn_model.keras")
    model.save(model_path)

    p_val = model.predict([X_val_i, m_val], verbose=0).reshape(-1)
    return model_path, float(mae(y_val_i, p_val))


def main():
    args = build_argparser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = load_train_df(args.data_dir)

    X, y = make_feature_matrix(df, args.data_dir, args.img_size)

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    scaler, svm, mlp = train_classical_models(X_tr, y_tr)
    p_svm_va, p_mlp_va = predict_classical(scaler, svm, mlp, X_va)

    w = fit_fusion_weights(y_va, p_svm_va, p_mlp_va)
    p_fuse_va = apply_fusion(w, p_svm_va, p_mlp_va)

    results = {
        "val_mae_svm": mae(y_va, p_svm_va),
        "val_mae_mlp": mae(y_va, p_mlp_va),
        "val_mae_fusion": mae(y_va, p_fuse_va),
        "val_acc_±12m_fusion": tolerance_accuracy_months(y_va, p_fuse_va, 12),
        "val_acc_±24m_fusion": tolerance_accuracy_months(y_va, p_fuse_va, 24),
        "fusion_weights": {"w_svm": float(w[0]), "w_mlp": float(w[1]), "bias": float(w[2])},
    }

    if args.use_cnn:
        model_path, cnn_mae = train_optional_cnn(df, args.data_dir, args.img_size, args.out_dir)
        results["cnn_model_path"] = model_path
        results["val_mae_cnn"] = cnn_mae

    joblib.dump(scaler, os.path.join(args.out_dir, "scaler.joblib"))
    joblib.dump(svm, os.path.join(args.out_dir, "svm.joblib"))
    joblib.dump(mlp, os.path.join(args.out_dir, "mlp.joblib"))
    joblib.dump(w, os.path.join(args.out_dir, "fusion_weights.joblib"))

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Training complete. Saved to:", args.out_dir)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

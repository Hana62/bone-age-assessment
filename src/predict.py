import os
import argparse
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocessing import preprocess_image, extract_rois
from features import extract_features


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True,
                   help="Folder with test.csv and test/ images")
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--artifacts_dir", type=str, required=True,
                   help="Folder created by train.py (scaler/svm/mlp/fusion weights)")
    p.add_argument("--out_csv", type=str, default="submission.csv")
    return p


def load_test_df(data_dir: str) -> pd.DataFrame:
    test_csv = os.path.join(data_dir, "test.csv")
    df = pd.read_csv(test_csv)
    if "id" not in df.columns:
        raise ValueError("test.csv must contain an 'id' column.")
    if "male" not in df.columns:
        df["male"] = 0
    return df


def make_X_test(df: pd.DataFrame, data_dir: str, img_size: int):
    X_list = []
    ids = []

    img_folder = os.path.join(data_dir, "test")
    if not os.path.isdir(img_folder):
        raise FileNotFoundError(f"Expected folder: {img_folder}")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting test features"):
        img_id = row["id"]
        path = os.path.join(img_folder, f"{img_id}.png")

        img = preprocess_image(path, out_size=(img_size, img_size))
        rois = extract_rois(img)
        feats = extract_features(rois)

        male = float(row.get("male", 0))
        feats = np.concatenate([feats, np.array([male], dtype=np.float32)], axis=0)

        X_list.append(feats)
        ids.append(int(img_id))

    X = np.vstack(X_list).astype(np.float32)
    return ids, X


def main():
    args = build_argparser().parse_args()

    scaler = joblib.load(os.path.join(args.artifacts_dir, "scaler.joblib"))
    svm = joblib.load(os.path.join(args.artifacts_dir, "svm.joblib"))
    mlp = joblib.load(os.path.join(args.artifacts_dir, "mlp.joblib"))
    w = joblib.load(os.path.join(args.artifacts_dir, "fusion_weights.joblib"))

    df_test = load_test_df(args.data_dir)
    ids, X = make_X_test(df_test, args.data_dir, args.img_size)

    Xs = scaler.transform(X)
    p_svm = svm.predict(Xs)
    p_mlp = mlp.predict(Xs)
    pred = w[0] * p_svm + w[1] * p_mlp + w[2]

    sub = pd.DataFrame({"id": ids, "boneage": pred})
    sub.to_csv(args.out_csv, index=False)
    print("Wrote:", args.out_csv)


if __name__ == "__main__":
    main()

"""
Multi-label training for NIH ChestX-ray14 (CheXNet-style).
=========================================================
Khác với train.py (3 lớp softmax), file này train **14 nhãn độc lập** bằng
sigmoid + binary_crossentropy (đúng bản chất NIH dataset).

Khuyến nghị chạy trên Google Colab GPU T4.

Quick start (Colab):
    !pip install -q tensorflow pandas scikit-learn
    # Mount drive, đặt CSV + ảnh vào DATA_DIR
    !python train_multilabel.py --epochs 10 --samples-per-class 1000

Output:
    saved_model/xray_chexnet14.h5
    saved_model/training_history.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]
NUM_CLASSES = len(LABELS)
IMG_SIZE = 224


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data", help="Thư mục chứa images_*/")
    p.add_argument("--csv", default="data/Data_Entry_2017.csv")
    p.add_argument("--output", default="saved_model")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--samples-per-class", type=int, default=1000, help="Mỗi nhãn lấy tối đa bao nhiêu ảnh dương tính")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--fine-tune-epochs", type=int, default=8)
    p.add_argument("--fine-tune-lr", type=float, default=1e-5)
    p.add_argument("--unfreeze-last", type=int, default=80)
    return p.parse_args()


def build_index(csv_path: str, data_dir: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Tạo bản đồ filename → đường dẫn (ảnh nằm trong images_001..012/images/)
    paths: dict[str, str] = {}
    for sub in sorted(Path(data_dir).glob("images_*/images")):
        for f in sub.iterdir():
            if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                paths[f.name] = str(f)

    df["path"] = df["Image Index"].map(paths)
    df = df.dropna(subset=["path"]).reset_index(drop=True)

    findings = df["Finding Labels"].fillna("").str.split("|")
    for label in LABELS:
        df[label] = findings.apply(lambda s, lb=label: int(lb in s))

    return df


def balance_subset(df: pd.DataFrame, samples_per_class: int) -> pd.DataFrame:
    """Lấy subset cân bằng: với mỗi nhãn, sample tối đa N ảnh dương tính."""
    pos_indices = set()
    for label in LABELS:
        idx = df.index[df[label] == 1].tolist()
        if len(idx) > samples_per_class:
            idx = list(np.random.RandomState(42).choice(idx, samples_per_class, replace=False))
        pos_indices.update(idx)

    # Thêm "No Finding" để mô hình học cả ảnh khoẻ
    no_finding = df.index[df["Finding Labels"] == "No Finding"].tolist()
    if len(no_finding) > samples_per_class:
        no_finding = list(np.random.RandomState(42).choice(no_finding, samples_per_class, replace=False))
    pos_indices.update(no_finding)

    return df.loc[sorted(pos_indices)].reset_index(drop=True)


def make_dataset(df: pd.DataFrame, batch_size: int, training: bool) -> tf.data.Dataset:
    paths = df["path"].values
    labels = df[LABELS].values.astype("float32")

    def _load(path, lbl):
        img = tf.io.read_file(path)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        img = preprocess_input(tf.cast(img, tf.float32))
        return img, lbl

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(2048, reshuffle_each_iteration=True)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(
            lambda x, y: (tf.image.random_flip_left_right(x), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model() -> Model:
    base = DenseNet121(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.4)(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(NUM_CLASSES, activation="sigmoid")(x)
    model = Model(base.input, out)
    return model


def compile_model(model: Model, lr: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc", multi_label=True, num_labels=NUM_CLASSES),
            tf.keras.metrics.BinaryAccuracy(name="acc", threshold=0.5),
        ],
    )


def main() -> None:
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    df = build_index(args.csv, args.data_dir)
    df = balance_subset(df, args.samples_per_class)
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    train_ds = make_dataset(train_df, args.batch_size, training=True)
    val_ds = make_dataset(val_df, args.batch_size, training=False)

    model = build_model()
    compile_model(model, args.lr)
    ckpt_path = os.path.join(args.output, "xray_chexnet14.h5")
    callbacks = [
        EarlyStopping(monitor="val_auc", mode="max", patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
        ModelCheckpoint(ckpt_path, monitor="val_auc", mode="max", save_best_only=True),
    ]

    print("\n🚀 Phase 1: Training classification head...")
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    print(f"\n🔓 Phase 2: Fine-tuning last {args.unfreeze_last} layers...")
    base = model.layers[1] if isinstance(model.layers[0], tf.keras.layers.InputLayer) else model.layers[0]
    base.trainable = True
    for layer in base.layers[: -args.unfreeze_last]:
        layer.trainable = False
    compile_model(model, args.fine_tune_lr)
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs, callbacks=callbacks)

    history = {
        "phase1": {k: [float(v) for v in vs] for k, vs in h1.history.items()},
        "phase2": {k: [float(v) for v in vs] for k, vs in h2.history.items()},
        "labels": LABELS,
    }
    with open(os.path.join(args.output, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\n✅ Saved best model to {ckpt_path}")
    print(f"✅ Saved training history to {args.output}/training_history.json")
    print("Tip: dùng MODEL_TYPE=auto + MODEL_PATH=saved_model/xray_chexnet14.h5; backend tự nhận 14 lớp.")


if __name__ == "__main__":
    main()

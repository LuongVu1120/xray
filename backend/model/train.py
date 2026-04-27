"""
Training script for AI X-Ray Diagnosis System
=============================================
Recommended: Run on Google Colab with GPU T4 (free)
https://colab.research.google.com

Steps:
1. Upload this file to Colab
2. Mount Google Drive to save the model
3. Download NIH Chest X-Ray dataset from:
   https://www.kaggle.com/datasets/nih-chest-xrays/data
4. Run this script
5. Download saved_model/xray_model.h5 and place in backend/saved_model/
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────
IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS      = 15
SAMPLES_PER_CLASS = 500   # Increase to 1500+ for better accuracy
DATA_DIR    = "data"       # Thư mục chứa ảnh NIH
CSV_PATH    = "data/Data_Entry_2017.csv"
OUTPUT_DIR  = "saved_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{DATA_DIR}/train/Normal", exist_ok=True)
os.makedirs(f"{DATA_DIR}/train/Pneumonia", exist_ok=True)
os.makedirs(f"{DATA_DIR}/train/Other", exist_ok=True)
os.makedirs(f"{DATA_DIR}/test/Normal", exist_ok=True)
os.makedirs(f"{DATA_DIR}/test/Pneumonia", exist_ok=True)
os.makedirs(f"{DATA_DIR}/test/Other", exist_ok=True)


# ── 1. Chuẩn bị dữ liệu ────────────────────────────────
def label_group(finding: str) -> str:
    if finding == "No Finding":
        return "Normal"
    elif "Pneumonia" in finding:
        return "Pneumonia"
    else:
        return "Other"


def prepare_data():
    print("📂 Reading CSV...")
    df = pd.read_csv(CSV_PATH)
    df["label"] = df["Finding Labels"].apply(label_group)

    # Lấy mẫu cân bằng
    sample = df.groupby("label").sample(SAMPLES_PER_CLASS, random_state=42)
    train_df, test_df = train_test_split(sample, test_size=0.2, stratify=sample["label"], random_state=42)

    print(f"✅ Train: {len(train_df)} | Test: {len(test_df)}")
    print(train_df["label"].value_counts())

    # Copy ảnh vào thư mục train/test
    import shutil
    all_image_dirs = [
        "images_001/images", "images_002/images", "images_003/images",
        "images_004/images", "images_005/images", "images_006/images",
        "images_007/images", "images_008/images", "images_009/images",
        "images_010/images", "images_011/images", "images_012/images",
    ]

    def find_image(filename):
        for d in all_image_dirs:
            path = os.path.join(DATA_DIR, d, filename)
            if os.path.exists(path):
                return path
        return None

    def copy_images(dataframe, split):
        for _, row in dataframe.iterrows():
            src = find_image(row["Image Index"])
            if src:
                dst = f"{DATA_DIR}/{split}/{row['label']}/{row['Image Index']}"
                shutil.copy2(src, dst)

    print("📋 Copying train images...")
    copy_images(train_df, "train")
    print("📋 Copying test images...")
    copy_images(test_df, "test")
    print("✅ Data preparation complete!")


# ── 2. Build Model ──────────────────────────────────────
def build_model(num_classes: int = 3) -> Model:
    base = DenseNet121(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    # Freeze base initially
    base.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ── 3. Data Generators ──────────────────────────────────
def get_generators():
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.9, 1.1],
        fill_mode="nearest"
    )
    test_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_data = train_gen.flow_from_directory(
        f"{DATA_DIR}/train",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True
    )
    test_data = test_gen.flow_from_directory(
        f"{DATA_DIR}/test",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )
    return train_data, test_data


# ── 4. Train ────────────────────────────────────────────
def train():
    prepare_data()
    model = build_model()

    train_data, test_data = get_generators()

    callbacks = [
        EarlyStopping(patience=4, restore_best_weights=True, monitor="val_accuracy"),
        ReduceLROnPlateau(factor=0.5, patience=2, monitor="val_loss"),
        ModelCheckpoint(f"{OUTPUT_DIR}/xray_model.h5", save_best_only=True, monitor="val_accuracy")
    ]

    print("\n🚀 Phase 1: Training classification head...")
    history1 = model.fit(
        train_data,
        validation_data=test_data,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Fine-tune: unfreeze last 50 layers
    print("\n🔓 Phase 2: Fine-tuning DenseNet layers...")
    base_model = model.layers[0]
    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    history2 = model.fit(
        train_data,
        validation_data=test_data,
        epochs=10,
        callbacks=callbacks
    )

    # Plot results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history1.history["accuracy"] + history2.history["accuracy"], label="Train")
    plt.plot(history1.history["val_accuracy"] + history2.history["val_accuracy"], label="Val")
    plt.title("Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history1.history["loss"] + history2.history["loss"], label="Train")
    plt.plot(history1.history["val_loss"] + history2.history["val_loss"], label="Val")
    plt.title("Loss")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/training_history.png")
    print(f"\n✅ Model saved to {OUTPUT_DIR}/xray_model.h5")
    print(f"✅ Training chart saved to {OUTPUT_DIR}/training_history.png")


if __name__ == "__main__":
    train()

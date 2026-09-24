"""
Model Training Script — สคริปต์สำหรับเทรนโมเดลตรวจหาเชื้อมาลาเรีย

ใช้ NIH Malaria Dataset และ MobileNetV2 transfer learning
สามารถรันแยกต่างหากเพื่อเทรนโมเดล แล้วบันทึกไว้ใช้กับแอป

Usage:
    python models/train_model.py --download    # ดาวน์โหลด dataset
    python models/train_model.py --train       # เทรนโมเดล
    python models/train_model.py --all         # ดาวน์โหลด + เทรน
"""

import os
import sys
import argparse
import zipfile
import shutil
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# --- Paths ---
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATASET_DIR = os.path.join(DATA_DIR, "cell_images")
MODELS_DIR = os.path.join(PROJECT_ROOT, "trained_models")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "parasite_detector.h5")

# --- Hyperparameters ---
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42


def download_dataset():
    """ดาวน์โหลด NIH Malaria Dataset

    Dataset: https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-datasets
    หรือจาก Kaggle: https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria
    """
    print("=" * 60)
    print("📥 ดาวน์โหลด NIH Malaria Dataset")
    print("=" * 60)

    os.makedirs(DATA_DIR, exist_ok=True)

    # Check if dataset already exists
    parasitized_dir = os.path.join(DATASET_DIR, "Parasitized")
    uninfected_dir = os.path.join(DATASET_DIR, "Uninfected")

    if os.path.exists(parasitized_dir) and os.path.exists(uninfected_dir):
        p_count = len(os.listdir(parasitized_dir))
        u_count = len(os.listdir(uninfected_dir))
        print(f"✅ Dataset มีอยู่แล้ว!")
        print(f"   Parasitized: {p_count:,} ภาพ")
        print(f"   Uninfected:  {u_count:,} ภาพ")
        return True

    print("\n⚠️  ไม่พบ Dataset ในเครื่อง")
    print("\n📋 วิธีดาวน์โหลด Dataset:")
    print("-" * 40)
    print("1. ไปที่: https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria")
    print("2. ดาวน์โหลดไฟล์ cell_images.zip")
    print(f"3. แตกไฟล์ไปที่: {DATASET_DIR}")
    print("4. โครงสร้างโฟลเดอร์ควรเป็น:")
    print(f"   {DATASET_DIR}/")
    print("   ├── Parasitized/    (ภาพเซลล์ที่ติดเชื้อ)")
    print("   └── Uninfected/     (ภาพเซลล์ปกติ)")
    print()

    # Try to use kaggle API if available
    try:
        import kaggle

        print("🔄 พบ Kaggle API — กำลังดาวน์โหลดอัตโนมัติ...")
        kaggle.api.dataset_download_files(
            "iarunava/cell-images-for-detecting-malaria",
            path=DATA_DIR,
            unzip=True,
        )
        print("✅ ดาวน์โหลดสำเร็จ!")
        return True
    except ImportError:
        print("ℹ️  ไม่พบ kaggle package — กรุณาดาวน์โหลดด้วยตนเอง")
        print("   ติดตั้ง kaggle: pip install kaggle")
        return False
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
        print("   กรุณาดาวน์โหลดด้วยตนเอง")
        return False


def create_model():
    """สร้างโมเดล CNN ด้วย MobileNetV2 transfer learning

    Returns:
        Compiled Keras model
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    print("\n🏗️  กำลังสร้างโมเดล...")

    # Base model: MobileNetV2 (lightweight, good for medical imaging)
    base_model = keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )

    # Freeze base model layers
    base_model.trainable = False

    # Build model
    model = keras.Sequential(
        [
            # Input preprocessing
            layers.InputLayer(input_shape=(*IMAGE_SIZE, 3)),
            # Data augmentation (only during training)
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.1),
            # Preprocessing for MobileNetV2
            layers.Rescaling(1.0 / 127.5, offset=-1),
            # Base model
            base_model,
            # Classification head
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),  # Binary: infected vs not
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )

    model.summary()
    return model


def load_dataset():
    """โหลด dataset และแบ่ง train/validation

    Returns:
        (train_dataset, val_dataset)
    """
    import tensorflow as tf

    print("\n📂 กำลังโหลด dataset...")

    parasitized_dir = os.path.join(DATASET_DIR, "Parasitized")
    uninfected_dir = os.path.join(DATASET_DIR, "Uninfected")

    if not os.path.exists(parasitized_dir) or not os.path.exists(uninfected_dir):
        raise FileNotFoundError(
            f"ไม่พบ dataset ที่ {DATASET_DIR}\n"
            "กรุณารัน: python models/train_model.py --download"
        )

    # Use tf.keras.utils.image_dataset_from_directory
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=RANDOM_SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=RANDOM_SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
    )

    class_names = train_ds.class_names
    print(f"✅ Classes: {class_names}")
    print(f"   Training batches:   {len(train_ds)}")
    print(f"   Validation batches: {len(val_ds)}")

    # Performance optimization
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds


def train_model():
    """เทรนโมเดลตรวจหาเชื้อมาลาเรีย"""
    import tensorflow as tf
    from tensorflow import keras

    print("=" * 60)
    print("🏋️  เทรนโมเดลตรวจหาเชื้อมาลาเรีย")
    print("=" * 60)
    print(f"   Image size:       {IMAGE_SIZE}")
    print(f"   Batch size:       {BATCH_SIZE}")
    print(f"   Epochs:           {EPOCHS}")
    print(f"   Learning rate:    {LEARNING_RATE}")
    print(f"   Validation split: {VALIDATION_SPLIT}")

    # Load data
    train_ds, val_ds = load_dataset()

    # Create model
    model = create_model()

    # Callbacks
    os.makedirs(MODELS_DIR, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=5,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor="val_auc",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(MODELS_DIR, "logs"),
            histogram_freq=1,
        ),
    ]

    # Train
    print("\n🚀 เริ่มเทรน...\n")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # Save final model
    model.save(MODEL_SAVE_PATH)
    print(f"\n✅ บันทึกโมเดลที่: {MODEL_SAVE_PATH}")

    # Print final metrics
    print("\n📊 ผลลัพธ์สุดท้าย:")
    print("-" * 40)
    val_metrics = {k: v[-1] for k, v in history.history.items() if k.startswith("val_")}
    for name, value in val_metrics.items():
        print(f"   {name}: {value:.4f}")

    # Plot training history
    try:
        _plot_training_history(history)
    except Exception as e:
        print(f"⚠️  ไม่สามารถสร้างกราฟ: {e}")

    return model, history


def _plot_training_history(history):
    """สร้างกราฟแสดงผลการเทรน"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history.history["accuracy"], label="Train Accuracy")
    axes[1].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # AUC
    axes[2].plot(history.history["auc"], label="Train AUC")
    axes[2].plot(history.history["val_auc"], label="Val AUC")
    axes[2].set_title("AUC")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = os.path.join(MODELS_DIR, "training_history.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"📈 บันทึกกราฟที่: {plot_path}")
    plt.close()


def evaluate_model(model_path: str = MODEL_SAVE_PATH):
    """ประเมินผลโมเดลที่เทรนแล้ว"""
    import tensorflow as tf
    from tensorflow import keras

    print("=" * 60)
    print("📊 ประเมินผลโมเดล")
    print("=" * 60)

    if not os.path.exists(model_path):
        print(f"❌ ไม่พบโมเดลที่: {model_path}")
        return

    model = keras.models.load_model(model_path)

    # Load validation data
    _, val_ds = load_dataset()

    # Evaluate
    results = model.evaluate(val_ds, verbose=1)

    print("\n📊 ผลการประเมิน:")
    print("-" * 40)
    for name, value in zip(model.metrics_names, results):
        print(f"   {name}: {value:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="🔬 เทรนโมเดลตรวจหาเชื้อมาลาเรีย"
    )
    parser.add_argument(
        "--download", action="store_true", help="ดาวน์โหลด NIH Malaria Dataset"
    )
    parser.add_argument(
        "--train", action="store_true", help="เทรนโมเดล"
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="ประเมินผลโมเดล"
    )
    parser.add_argument(
        "--all", action="store_true", help="ดาวน์โหลด + เทรน"
    )

    args = parser.parse_args()

    if args.all or args.download:
        download_dataset()

    if args.all or args.train:
        train_model()

    if args.evaluate:
        evaluate_model()

    if not any([args.download, args.train, args.evaluate, args.all]):
        parser.print_help()
        print("\n💡 ตัวอย่างการใช้งาน:")
        print("   python models/train_model.py --download  # ดาวน์โหลด dataset")
        print("   python models/train_model.py --train     # เทรนโมเดล")
        print("   python models/train_model.py --all       # ดาวน์โหลด + เทรน")


if __name__ == "__main__":
    main()

"""
Step 2 of 3 - Train the CNN.

What this script does, in plain English:
  1. Loads two sets of images:
        dataset/trained/      -> 500 photos the CNN learns from
            fresh/   (250)
            rotten/  (250)
        dataset/not_trained/  -> 300 photos kept aside for testing
            fresh/   (150)
            rotten/  (150)
  2. Resizes every image to 128 x 128 pixels and scales colors to 0..1.
  3. Builds a small Convolutional Neural Network (CNN) from scratch.
     The layers, in order, are:
        Conv2D(32)  -> MaxPool   "find edges and corners"
        Conv2D(64)  -> MaxPool   "find curves and dark spots"
        Conv2D(128) -> MaxPool   "find tomato-level patterns"
        Flatten -> Dense(128) -> Dropout -> Dense(1, sigmoid)
  4. Trains for ~15 epochs with light augmentation (flip, rotate, zoom).
  5. Tests on the 300 unseen images and prints accuracy.
  6. Saves:
        tomato_model.h5         (the trained model)
        training_results.json   (numbers used by training visualizations)

Run AFTER step 1. Usage:
    python 2_train_model.py
"""

import json
import os
import random
from pathlib import Path

import numpy as np

# Hide TensorFlow's noisy logs (cleaner output for students/teachers).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow.keras import layers, models

# --- Configuration --------------------------------------------------------- #

PROJECT_ROOT     = Path(__file__).resolve().parent
DATASET_DIR      = PROJECT_ROOT / "dataset"
TRAINED_DIR      = DATASET_DIR / "trained"
NOT_TRAINED_DIR  = DATASET_DIR / "not_trained"
MODEL_PATH       = PROJECT_ROOT / "tomato_model.h5"
RESULTS_PATH     = PROJECT_ROOT / "training_results.json"

IMG_SIZE     = 128         # all images resized to 128 x 128
BATCH_SIZE   = 32
EPOCHS       = 15
RANDOM_SEED  = 42

# --- Reproducibility ------------------------------------------------------- #

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# --- Data loading ---------------------------------------------------------- #

def load_folder(root: Path):
    """Load every .jpg under root/fresh and root/rotten.

    Returns (X, y, filenames):
        X         -> array (N, 128, 128, 3), pixel values in 0..1
        y         -> array (N,), 0 = fresh, 1 = rotten
        filenames -> list of N file names
    """
    classes = [("fresh", 0), ("rotten", 1)]
    paths, labels = [], []
    for name, label in classes:
        folder = root / name
        if not folder.exists():
            raise SystemExit(
                f"Folder not found: {folder}\n"
                "Run step 1 first:   python 1_prepare_dataset.py"
            )
        for img_path in sorted(folder.glob("*.jpg")):
            paths.append(img_path)
            labels.append(label)

    if not paths:
        raise SystemExit(f"No .jpg files found in {root}")

    X = np.zeros((len(paths), IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    for i, p in enumerate(paths):
        img = tf.keras.utils.load_img(str(p), target_size=(IMG_SIZE, IMG_SIZE))
        X[i] = tf.keras.utils.img_to_array(img) / 255.0

    y = np.array(labels, dtype=np.float32)
    files = [p.name for p in paths]
    return X, y, files


def shuffle_in_place(X, y, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    return X[idx], y[idx]


# --- Model ----------------------------------------------------------------- #

def build_cnn() -> tf.keras.Model:
    """A small CNN that any student can defend in a viva.

    Each layer's job is written in the inline comment.
    """
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

        # Light data augmentation -- only active during training.
        # Helps the model see "more" data than 500 images, by randomly
        # flipping / rotating / zooming each training image.
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),

        # Block 1 -- find edges and corners.
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        # Block 2 -- find curves and dark spots.
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        # Block 3 -- find tomato-level patterns (mold patches, full outline).
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        # Turn the image features into a single long list of numbers.
        layers.Flatten(),

        # Mix everything together with a fully-connected layer.
        layers.Dense(128, activation="relu"),

        # Dropout: during training, randomly ignore 50% of the neurons.
        # Forces the model to NOT memorize specific images.
        layers.Dropout(0.5),

        # Output: a number between 0 (fresh) and 1 (rotten).
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# --- Evaluation ------------------------------------------------------------ #

def confusion_counts(y_true, y_pred) -> dict:
    """Count TP/TN/FP/FN for binary labels."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))   # rotten predicted rotten
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))   # fresh  predicted fresh
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))   # fresh  predicted rotten
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))   # rotten predicted fresh
    return {
        "true_fresh_predicted_fresh":   tn,
        "true_fresh_predicted_rotten":  fp,
        "true_rotten_predicted_fresh":  fn,
        "true_rotten_predicted_rotten": tp,
    }


# --- Main ------------------------------------------------------------------ #

def main() -> None:
    print("=" * 60)
    print("TomatoGuard - Step 2 of 3:  Train CNN")
    print("=" * 60)

    print(f"[..] Loading TRAINED set from {TRAINED_DIR.relative_to(PROJECT_ROOT)} ...")
    X_tr, y_tr, _ = load_folder(TRAINED_DIR)
    X_tr, y_tr = shuffle_in_place(X_tr, y_tr, RANDOM_SEED)
    print(f"     {len(X_tr)} images "
          f"({int((y_tr==0).sum())} fresh, {int((y_tr==1).sum())} rotten).")

    print(f"[..] Loading NOT_TRAINED (test) set from "
          f"{NOT_TRAINED_DIR.relative_to(PROJECT_ROOT)} ...")
    X_te, y_te, _ = load_folder(NOT_TRAINED_DIR)
    print(f"     {len(X_te)} images "
          f"({int((y_te==0).sum())} fresh, {int((y_te==1).sum())} rotten).")

    print("[..] Building the CNN ...")
    model = build_cnn()
    model.summary()

    print(f"[..] Training for {EPOCHS} epochs ...")
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_te, y_te),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
    )

    print("[..] Final evaluation on the not_trained test set ...")
    test_loss, test_acc = model.evaluate(X_te, y_te, verbose=0)
    print(f"     Test accuracy: {test_acc*100:.2f}%   Test loss: {test_loss:.4f}")

    probs = model.predict(X_te, verbose=0).ravel()
    preds = (probs >= 0.5).astype(np.float32)
    conf = confusion_counts(y_te, preds)
    print("     Confusion matrix:")
    print(f"       fresh  predicted fresh  : {conf['true_fresh_predicted_fresh']}")
    print(f"       fresh  predicted rotten : {conf['true_fresh_predicted_rotten']}")
    print(f"       rotten predicted fresh  : {conf['true_rotten_predicted_fresh']}")
    print(f"       rotten predicted rotten : {conf['true_rotten_predicted_rotten']}")

    print(f"[..] Saving model to {MODEL_PATH.name} ...")
    model.save(MODEL_PATH)

    print(f"[..] Saving training_results.json ...")
    results = {
        "image_size": IMG_SIZE,
        "epochs": EPOCHS,
        "train_count": int(len(X_tr)),
        "test_count":  int(len(X_te)),
        "history": {
            "accuracy":      [float(v) for v in history.history["accuracy"]],
            "val_accuracy":  [float(v) for v in history.history["val_accuracy"]],
            "loss":          [float(v) for v in history.history["loss"]],
            "val_loss":      [float(v) for v in history.history["val_loss"]],
        },
        "test_accuracy": float(test_acc),
        "test_loss":     float(test_loss),
        "confusion_matrix": conf,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))

    print()
    print("[done] Model trained.")
    print(f"       Model file:    {MODEL_PATH.name}")
    print(f"       Results file:  {RESULTS_PATH.name}")
    print()
    print("Next step:  python 3_server.py")


if __name__ == "__main__":
    main()

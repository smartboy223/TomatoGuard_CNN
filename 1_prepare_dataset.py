"""
Step 1 of 3 - Prepare the tomato dataset.

What this script does, in plain English:
  1. Downloads the FGrade tomato dataset from GitHub (one time only).
     FGrade has 6,470 photos of single tomatoes, labeled 1 to 10
     where 1 = freshest and 10 = most rotten.
  2. Following the dataset paper's own 2-class split:
        classes 1 to 5  -> "fresh"
        classes 6 to 10 -> "rotten"
  3. Picks 800 photos total (400 fresh + 400 rotten), then splits them
     into TWO clearly named folders so anyone looking at the project
     can see at a glance what the CNN learned from and what it didn't:

        dataset/trained/      <- 500 images the CNN learns from
            fresh/    (250)
            rotten/   (250)

        dataset/not_trained/  <- 300 images the CNN has NEVER seen
            fresh/    (150)        (used only to test the model)
            rotten/   (150)

Run this ONCE before training. Re-running rebuilds the folders cleanly.

Usage:
    python 1_prepare_dataset.py
"""

import random
import shutil
import subprocess
import sys
from pathlib import Path

# --- Configuration --------------------------------------------------------- #

PROJECT_ROOT  = Path(__file__).resolve().parent
DATASET_DIR   = PROJECT_ROOT / "dataset"
TRAINED_DIR   = DATASET_DIR / "trained"
NOT_TRAINED_DIR = DATASET_DIR / "not_trained"

TRAINED_PER_CLASS     = 250    # 250 fresh + 250 rotten = 500 in trained/
NOT_TRAINED_PER_CLASS = 150    # 150 fresh + 150 rotten = 300 in not_trained/
RANDOM_SEED           = 42     # seeded so re-runs give the same split

FGRADE_REPO_URL   = "https://github.com/skarifahmed/FGrade.git"
FGRADE_CACHE_DIR  = PROJECT_ROOT / ".fgrade_cache"

# FGrade folder names are "0".."9" (their paper calls them Class 1..10).
FRESH_FOLDERS   = ["0", "1", "2", "3", "4"]   # Class 1-5 -> fresh
ROTTEN_FOLDERS  = ["5", "6", "7", "8", "9"]   # Class 6-10 -> rotten


# --- Helpers --------------------------------------------------------------- #

def clone_fgrade_if_needed() -> Path:
    """Clone the FGrade repo into .fgrade_cache/ if it isn't already there."""
    if FGRADE_CACHE_DIR.exists() and any(FGRADE_CACHE_DIR.iterdir()):
        print(f"[ok] FGrade already downloaded at {FGRADE_CACHE_DIR}")
        return FGRADE_CACHE_DIR

    print(f"[..] Downloading FGrade tomato dataset from GitHub...")
    print(f"     (about 200 MB, one-time download)")
    FGRADE_CACHE_DIR.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", FGRADE_REPO_URL, str(FGRADE_CACHE_DIR)],
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print("\n[!!] Could not download FGrade automatically.")
        print("     You need 'git' installed and an internet connection.")
        print("     Manual fix: download a ZIP from")
        print(f"       {FGRADE_REPO_URL.replace('.git','')}")
        print(f"     and extract it into:  {FGRADE_CACHE_DIR}")
        print(f"\n     Error was: {e}")
        sys.exit(1)
    print(f"[ok] Downloaded to {FGRADE_CACHE_DIR}")
    return FGRADE_CACHE_DIR


def collect_images(fgrade_root: Path, class_folders: list) -> list:
    """Walk a list of FGrade class folders and return every .jpg path."""
    images = []
    for sub in ("Training_set", "Testing_set"):
        for cls in class_folders:
            folder = fgrade_root / "data" / sub / cls
            if folder.exists():
                images.extend(sorted(folder.glob("*.jpg")))
    return images


def write_split(src_paths, label, trained_count, not_trained_count):
    """Pick (trained_count + not_trained_count) images and put them into the
    two output folders (trained/<label>/  and  not_trained/<label>/).
    Uses a seeded RNG so the split is reproducible.
    """
    needed = trained_count + not_trained_count
    if len(src_paths) < needed:
        raise SystemExit(
            f"[!!] Need {needed} {label} images but only {len(src_paths)} found."
        )

    rng = random.Random(RANDOM_SEED + (0 if label == "fresh" else 1))
    picks = rng.sample(src_paths, needed)

    trained_picks      = picks[:trained_count]
    not_trained_picks  = picks[trained_count:]

    # trained/
    out_t = TRAINED_DIR / label
    if out_t.exists(): shutil.rmtree(out_t)
    out_t.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(trained_picks, 1):
        shutil.copy2(src, out_t / f"{label}_{i:03d}.jpg")

    # not_trained/
    out_n = NOT_TRAINED_DIR / label
    if out_n.exists(): shutil.rmtree(out_n)
    out_n.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(not_trained_picks, 1):
        shutil.copy2(src, out_n / f"{label}_{i:03d}.jpg")

    print(f"[ok] {label:6s} -> trained: {trained_count}   "
          f"not_trained: {not_trained_count}")


# --- Main ------------------------------------------------------------------ #

def main() -> None:
    print("=" * 60)
    print("TomatoGuard - Step 1 of 3:  Prepare dataset")
    print("=" * 60)

    fgrade_root = clone_fgrade_if_needed()

    print("\n[..] Collecting fresh tomato photos (FGrade classes 1-5)...")
    fresh_images  = collect_images(fgrade_root, FRESH_FOLDERS)
    print(f"     Found {len(fresh_images)} fresh candidates.")

    print("[..] Collecting rotten tomato photos (FGrade classes 6-10)...")
    rotten_images = collect_images(fgrade_root, ROTTEN_FOLDERS)
    print(f"     Found {len(rotten_images)} rotten candidates.")

    # Remove old structure if present
    old_fresh  = DATASET_DIR / "fresh"
    old_rotten = DATASET_DIR / "rotten"
    if old_fresh.exists():  shutil.rmtree(old_fresh)
    if old_rotten.exists(): shutil.rmtree(old_rotten)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    print()
    write_split(fresh_images,  "fresh",  TRAINED_PER_CLASS, NOT_TRAINED_PER_CLASS)
    write_split(rotten_images, "rotten", TRAINED_PER_CLASS, NOT_TRAINED_PER_CLASS)

    print()
    print("[done] Dataset is ready.")
    print(f"       dataset/trained/      (500 photos the model LEARNS FROM)")
    print(f"       dataset/not_trained/  (300 photos kept aside for TESTING)")
    print()
    print("Next step:  python 2_train_model.py")


if __name__ == "__main__":
    main()

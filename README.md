# 🍅 TomatoGuard CNN

TomatoGuard is a local web demo for classifying tomato images as **fresh** or
**rotten** with a small Convolutional Neural Network (CNN). The project includes
the dataset preparation script, training script, saved TensorFlow model, local
Python server, and a browser UI for uploading a tomato image and viewing the
prediction.

✅ **The trained model is included in this repository as `tomato_model.h5`, so
anyone can clone the project and test the app immediately without retraining.**

This public version intentionally excludes private academic submission files,
personal identity details, and institution details.

## 📸 Screenshots

Real screenshots captured from the local HTML interface are stored in
`docs/screenshots/`.

![TomatoGuard home screen](docs/screenshots/tomatoguard-home.png)

![TomatoGuard prediction screen](docs/screenshots/tomatoguard-prediction.png)

## 🚀 What the Project Does

1. Downloads the public FGrade tomato image dataset.
2. Builds a balanced binary dataset:
   - `fresh`
   - `rotten`
3. Trains a small CNN from scratch.
4. Saves the trained TensorFlow model as `tomato_model.h5`.
5. Runs a local web app at `http://127.0.0.1:8000`.
6. Lets the user upload a tomato image and returns:
   - predicted label
   - fresh score
   - rotten score
   - confidence value

## 🍅 Dataset Source

The dataset source is **FGrade**, a public tomato freshness dataset by Das, Kar,
and Sekh. It contains 6,470 single-tomato images with freshness grades from 1 to
10.

Source repository: <https://github.com/skarifahmed/FGrade>

For this binary classifier, grades 1-5 are grouped as **fresh** and grades 6-10
are grouped as **rotten**. The project samples 800 images total:

| Split | Fresh | Rotten | Total |
| --- | ---: | ---: | ---: |
| `dataset/trained/` | 250 | 250 | 500 |
| `dataset/not_trained/` | 150 | 150 | 300 |
| Total | 400 | 400 | 800 |

## 🧠 Training Method

This project uses **supervised CNN image classification**, not Random Forest.
Random Forest is usually stronger for table-style feature data, while this
project works directly with images, so a Convolutional Neural Network is the
right fit.

Training details:

| Item | Value |
| --- | --- |
| Model type | Convolutional Neural Network |
| Framework | TensorFlow / Keras |
| Input size | 96 x 96 RGB images |
| Classes | `fresh`, `rotten` |
| Training images | 500 |
| Held-out test images | 300 |
| Epochs in saved run | 8 |
| Optimizer | Adam |
| Loss | Binary cross-entropy |
| Augmentation | Random flip, random rotation, random zoom |

## 📊 Current Model Accuracy and Scores

These numbers come from the uploaded `training_results.json` and the included
`tomato_model.h5` checkpoint.

| Metric | Score |
| --- | ---: |
| Final test accuracy | 60.00% |
| Best validation accuracy during training | 74.67% |
| Final test loss | 0.6271 |
| Fresh precision | 55.81% |
| Fresh recall | 96.00% |
| Fresh F1 score | 70.59% |
| Fresh F3 score | 89.55% |
| Fresh F5 score | 93.41% |
| Rotten precision | 85.71% |
| Rotten recall | 24.00% |
| Rotten F1 score | 37.50% |
| Rotten F3 score | 25.86% |
| Rotten F5 score | 24.68% |
| Macro F1 score | 54.04% |

Confusion matrix on the 300 held-out test images:

| Actual / Predicted | Fresh | Rotten |
| --- | ---: | ---: |
| Fresh | 144 | 6 |
| Rotten | 114 | 36 |

The model is good at recognizing fresh tomatoes in this saved run, but it misses
many rotten tomatoes. That makes it a useful portfolio and learning project,
while also showing a clear next improvement area: more rotten examples, stronger
augmentation, longer training, or transfer learning.

F3 and F5 are F-beta scores. They give more weight to recall than precision,
which is useful when the main question is whether the model catches as many
examples of a class as possible.

## 🏗️ Model Architecture

The model is a compact CNN trained from scratch:

```text
Input image
RandomFlip
RandomRotation
RandomZoom
Conv2D(32) + MaxPooling2D
Conv2D(64) + MaxPooling2D
Conv2D(128) + MaxPooling2D
Flatten
Dense(128)
Dropout(0.5)
Dense(1, sigmoid)
```

The final sigmoid output is interpreted as:

- below `0.5`: fresh
- `0.5` or above: rotten

Current saved training results are in `training_results.json`.

## 📁 Project Structure

```text
Final_tomatoes_cnn/
├── 1_prepare_dataset.py       # Download and prepare the FGrade image split
├── 2_train_model.py           # Train and evaluate the CNN
├── 3_server.py                # Run the local web server and prediction API
├── requirements.txt           # Python dependency list
├── start.bat                  # Windows launcher
├── tomato_colab.ipynb         # Optional Google Colab training notebook
├── tomato_model.h5            # Saved trained model
├── training_results.json      # Saved accuracy/loss/confusion matrix
├── assets/
│   └── background.png
├── dataset/
│   ├── trained/
│   │   ├── fresh/
│   │   └── rotten/
│   └── not_trained/
│       ├── fresh/
│       └── rotten/
├── docs/
│   ├── PROJECT_ARCHIVE.md
│   └── screenshots/
└── web/
    ├── index.html
    ├── style.css
    └── app.js
```

## ⚡ Easy Setup and Run

Use Python 3.9 or newer.

```powershell
git clone https://github.com/smartboy223/TomatoGuard_CNN.git
cd TomatoGuard_CNN
pip install -r requirements.txt
python 3_server.py
```

Then open:

```text
http://127.0.0.1:8000
```

The trained model and prepared dataset are already included. To rebuild the
dataset and retrain the model from scratch, run:

```powershell
pip install -r requirements.txt
python 1_prepare_dataset.py
python 2_train_model.py
python 3_server.py
```

On Windows, `start.bat` can also be used to launch the server after dependencies
are installed.

## 🖥️ How to Use

1. Start the server with `python 3_server.py`.
2. Open `http://127.0.0.1:8000`.
3. Click the upload area or drag a tomato image into it.
4. Click **Analyze Image**.
5. Review the predicted class and confidence scores.

## ⚠️ Notes for Public Use

- The app is designed for local demonstration and portfolio presentation.
- It is not a production food-safety tool.
- Predictions depend on image quality, lighting, background, and how similar the
  uploaded image is to the FGrade dataset.
- Private college report files are not part of this public repository.

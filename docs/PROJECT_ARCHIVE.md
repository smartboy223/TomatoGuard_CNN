# Project Archive

## Purpose

TomatoGuard is a portfolio-ready local CNN project that demonstrates a complete
image classification workflow: dataset preparation, model training, local model
serving, and a browser interface for predictions.

The trained model is included as `tomato_model.h5`, so the app can be tested
without retraining first.

The public archive removes private academic submission files, personal identity
details, and institution details. The original local state was preserved in a
separate snapshot before cleanup.

## Dataset

The project uses the public FGrade tomato dataset:

<https://github.com/skarifahmed/FGrade>

The source dataset contains 6,470 tomato images graded from 1 to 10. For this
project, grades 1-5 are treated as fresh tomatoes and grades 6-10 are treated as
rotten tomatoes.

The prepared project includes:

| Folder | Purpose | Count |
| --- | --- | ---: |
| `dataset/trained/fresh` | Training images | 250 |
| `dataset/trained/rotten` | Training images | 250 |
| `dataset/not_trained/fresh` | Held-out test images | 150 |
| `dataset/not_trained/rotten` | Held-out test images | 150 |

## Build Steps

Install dependencies:

```powershell
pip install -r requirements.txt
```

Recreate the dataset split:

```powershell
python 1_prepare_dataset.py
```

Train the CNN:

```powershell
python 2_train_model.py
```

Run the local app:

```powershell
python 3_server.py
```

Open:

```text
http://127.0.0.1:8000
```

## Training Method and Scores

The saved model was trained with supervised CNN image classification using
TensorFlow/Keras. It is not a Random Forest model.

Current saved run:

| Metric | Score |
| --- | ---: |
| Final test accuracy | 60.00% |
| Best validation accuracy | 74.67% |
| Final test loss | 0.6271 |
| Fresh F1 score | 70.59% |
| Fresh F3 score | 89.55% |
| Fresh F5 score | 93.41% |
| Rotten F1 score | 37.50% |
| Rotten F3 score | 25.86% |
| Rotten F5 score | 24.68% |
| Macro F1 score | 54.04% |

Confusion matrix:

| Actual / Predicted | Fresh | Rotten |
| --- | ---: | ---: |
| Fresh | 144 | 6 |
| Rotten | 114 | 36 |

## How the App Works

The local Python server serves the HTML/CSS/JavaScript files from `web/`. When a
user uploads an image, `web/app.js` sends it to `POST /predict`. The server
loads `tomato_model.h5`, resizes the image to the model input size, runs the
CNN, and returns a JSON prediction.

## Proof Screenshots

Screenshots are stored in `docs/screenshots/`:

- `tomatoguard-home.png`
- `tomatoguard-prediction.png`

They were captured from the local HTML interface running through the project
server.

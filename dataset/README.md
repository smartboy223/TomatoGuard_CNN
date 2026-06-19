# Dataset

This folder contains a sampled binary split from the public FGrade tomato
freshness dataset.

Source: <https://github.com/skarifahmed/FGrade>

The original dataset grades tomato freshness from 1 to 10. This project maps
grades 1-5 to `fresh` and grades 6-10 to `rotten`.

```text
dataset/
├── trained/
│   ├── fresh/   250 images used for training
│   └── rotten/  250 images used for training
└── not_trained/
    ├── fresh/   150 held-out images used for testing
    └── rotten/  150 held-out images used for testing
```

Run `python 1_prepare_dataset.py` to recreate the split from the upstream
dataset.

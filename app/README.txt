CANCER AI — STANDALONE APP (BM3, Rafay Nisar)
=============================================

WHAT THIS IS
  One app that predicts cancer from IMAGES and from FTIR SPECTRA (.xlsx/.csv),
  loading pre-trained model files. No datasets and no retraining needed.

PUT THESE FILES IN THIS FOLDER (next to Cancer_App_standalone.py)
  breakhis_model.pt     -> breast histopathology: benign / malignant     [from Kaggle app]
  sipakmed_model.pt     -> cervical cytology: 5 cell types                [from Kaggle app]
  ftir_pcalda.joblib    -> FTIR spectrum: cancer / PBMC                   [from FTIR notebook]
  best.pt               -> microscopy cell detection & count (OPTIONAL)   [from Colab detection notebook]

  (Only the tabs whose file is present will appear — so 3 files is enough.)

HOW TO RUN
  Option A - on your computer (needs Python 3.9+):
    1) open a terminal/command prompt in this folder
    2) pip install -r requirements.txt
    3) python Cancer_App_standalone.py
    4) open the link it prints (http://127.0.0.1:7860)

  Option B - in Google Colab (no install on your PC):
    1) upload this folder's files to a Colab session (left file panel)
    2) run:  !pip -q install -r requirements.txt
    3) run:  !python Cancer_App_standalone.py
    4) open the public share link it prints

HOW TO DEMO
  - Breast tab:    drop a BreakHis image (e.g. from a 'malignant' folder) -> benign/malignant + confidence
  - Cytology tab:  drop a SIPaKMeD cell image -> one of 5 classes + confidence
  - FTIR tab:      upload a *-raw.xlsx spectrum file -> cancer/PBMC proportion + verdict
  - Detection tab: (if best.pt present) drop a microscopy image -> boxes + cell count

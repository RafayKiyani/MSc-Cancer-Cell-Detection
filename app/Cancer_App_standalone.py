"""
Cancer_App_standalone.py — one app, runs from SAVED model files (no datasets, no retraining).
Predicts on IMAGES and on EXCEL/CSV SPECTRA.

Put the model files in the SAME folder as this script (only the ones you have are used):
    breakhis_model.pt     ResNet-50, benign/malignant           [Kaggle app]
    sipakmed_model.pt     ResNet-50, 5 cervical classes          [Kaggle app]
    ftir_pcalda.joblib    PCA-LDA cancer/PBMC + wavenumber grid  [FTIR notebook]
    best.pt               YOLO cancer-cell detector (optional)   [Colab detection notebook]

Install once:
    pip install -r requirements.txt
Run:
    python Cancer_App_standalone.py
Works on CPU. Only the tabs whose model file is present will appear.
"""
import os, numpy as np, torch, torch.nn as nn, torchvision
from torchvision import transforms
from PIL import Image
import gradio as gr

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
BH_CLASSES = ['benign', 'malignant']
SK_CLASSES = ['Dyskeratotic', 'Koilocytotic', 'Metaplastic', 'Parabasal', 'Superficial-Intermediate']

# ---------------- image classifiers ----------------
def load_resnet(path, classes):
    m = torchvision.models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(classes))
    m.load_state_dict(torch.load(path, map_location=dev)); m.to(dev).eval()
    print('loaded', path); return m

def classify(model, classes, img):
    if img is None: return {}
    x = tf(img.convert('RGB')).unsqueeze(0).to(dev)
    with torch.no_grad():
        p = torch.softmax(model(x), 1)[0].cpu().numpy()
    return {classes[i]: float(p[i]) for i in range(len(classes))}

bh = load_resnet('breakhis_model.pt', BH_CLASSES) if os.path.exists('breakhis_model.pt') else None
sk = load_resnet('sipakmed_model.pt', SK_CLASSES) if os.path.exists('sipakmed_model.pt') else None

# ---------------- FTIR spectrum classifier ----------------
ftir = None
if os.path.exists('ftir_pcalda.joblib'):
    try:
        import joblib
        ftir = joblib.load('ftir_pcalda.joblib'); print('loaded ftir_pcalda.joblib')
    except Exception as e:
        print('FTIR model not loaded:', e)

def predict_spectrum(file):
    if file is None or ftir is None:
        return {}, 'FTIR model (ftir_pcalda.joblib) not available.'
    import pandas as pd
    from scipy.signal import savgol_filter
    path = file.name if hasattr(file, 'name') else file
    df = pd.read_excel(path, header=None) if str(path).lower().endswith(('xlsx', 'xls')) else pd.read_csv(path, header=None)
    wn = pd.to_numeric(df.iloc[0, 1:], errors='coerce').to_numpy()
    X = df.iloc[1:, 1:].apply(pd.to_numeric, errors='coerce').to_numpy()
    keep = ~np.isnan(wn); wn = wn[keep]; X = np.nan_to_num(X[:, keep])
    grid = ftir['grid']; o = np.argsort(wn)
    Xr = np.vstack([np.interp(grid, wn[o], x[o]) for x in X])
    Xs = savgol_filter(Xr, 11, 2, axis=1); Xs = (Xs - Xs.mean(1, keepdims=True)) / (Xs.std(1, keepdims=True) + 1e-8)
    pred = ftir['model'].predict(Xs); n = len(pred); ncan = int((pred == 1).sum())
    label = {'Cancer': ncan / n, 'PBMC': (n - ncan) / n}
    msg = f'{n} spectra analysed — {ncan} cancer, {n-ncan} PBMC. Overall: {"CANCER-dominant" if ncan > n/2 else "CONTROL-dominant"}.'
    return label, msg

# ---------------- optional detector ----------------
yolo = None
if os.path.exists('best.pt'):
    try:
        from ultralytics import YOLO
        yolo = YOLO('best.pt'); print('loaded best.pt (detector)')
    except Exception as e:
        print('detector not loaded:', e)

def detect(img):
    if img is None or yolo is None:
        return None, 'Detector (best.pt) not available.'
    r = yolo.predict(img, conf=0.25, verbose=False)[0]
    return Image.fromarray(r.plot()[:, :, ::-1]), f'{len(r.boxes)} cancer cell(s) detected.'

# ---------------- interface ----------------
with gr.Blocks(title='Cancer AI - BM3') as app:
    gr.Markdown('# Cancer AI (BM3) — standalone demo\nUpload an **image** or an **FTIR spectrum (.xlsx/.csv)** on the matching tab. Runs from saved models; no dataset needed.')
    if bh is not None:
        with gr.Tab('Breast histopathology (benign/malignant)'):
            i = gr.Image(type='pil', label='Histopathology image')
            gr.Button('Classify', variant='primary').click(lambda x: classify(bh, BH_CLASSES, x), i, gr.Label(num_top_classes=2, label='Prediction'))
    if sk is not None:
        with gr.Tab('Cervical cytology (5 cell types)'):
            i = gr.Image(type='pil', label='Single-cell image')
            gr.Button('Classify', variant='primary').click(lambda x: classify(sk, SK_CLASSES, x), i, gr.Label(num_top_classes=5, label='Prediction'))
    if ftir is not None:
        with gr.Tab('FTIR spectrum (cancer/PBMC)'):
            f = gr.File(label='Spectrum file (.xlsx / .csv) — row 0 = wavenumbers')
            lab = gr.Label(label='Proportion'); txt = gr.Textbox(label='Result')
            gr.Button('Classify', variant='primary').click(predict_spectrum, f, [lab, txt])
    if yolo is not None:
        with gr.Tab('Microscopy cell detection & count'):
            i = gr.Image(type='pil', label='Microscopy image')
            o = gr.Image(label='Detections'); t = gr.Textbox(label='Result')
            gr.Button('Detect', variant='primary').click(detect, i, [o, t])
    if bh is None and sk is None and ftir is None and yolo is None:
        gr.Markdown('**No model files found.** Place the `.pt` / `.joblib` files next to this script.')

if __name__ == '__main__':
    app.launch(share=True)

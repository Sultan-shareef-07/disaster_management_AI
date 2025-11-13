[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/Sultan-shareef-07/disaster_management_AI/blob/main/disaster_management_ai_colab.ipynb)

# Disaster Management with AI — README

A compact, runnable mini-project that demonstrates how **AI augments an IoT disaster alert system**.
This repo contains:

* a **sensor anomaly** module (IsolationForest) that detects anomalous windows in time-series sensor data
* a **social-media classifier** (TF-IDF + Logistic Regression) that flags disaster-related tweets
* an **orchestrator (Flask)** endpoint that exposes model inference (text + sensor)
* a **Streamlit dashboard** to visualize sensor data and classify tweets interactively
* a **Colab notebook** that runs the whole demo on Google Colab (zero-install workflow)
* CI workflow that executes the notebook on GitHub Actions (optional)

Use this README to run locally (venv or conda) or on Google Colab, get outputs for your project report, and understand how pieces fit together.

---

## Repo layout (important files)

```
disaster-management/
├─ disaster_management_ai_colab.ipynb    # Colab demo notebook
├─ demo_data/
│  ├─ sensor_demo.csv
│  └─ tweets_demo.csv
├─ social_ml/
│  └─ src/
│     ├─ preprocess.py
│     ├─ train_model.py
│     ├─ predict.py
│     └─ explain.py
│  └─ models/
├─ cloud_ingest/
│  └─ sensor_model.py
├─ orchestrator/
│  ├─ api.py
│  └─ orchestrator.py
├─ dashboard/
│  └─ app_streamlit.py
├─ requirements.txt
├─ .github/workflows/run_notebook.yml   # (CI) runs the notebook and uploads outputs
└─ README.md
```

---

# Quick summary — how it works (conceptual)

1. **Sensor edge (NodeMCU hardware)** — pushes time-series sensor readings (vibration, flame, water) to the cloud (not implemented here; demo uses CSV).
2. **Sensor ML** (`cloud_ingest/sensor_model.py`) — featurizes non-overlapping windows (mean/std/max) and runs an IsolationForest to detect anomalous windows (possible events).
3. **Social ML** (`social_ml/src/train_model.py`, `predict.py`) — cleans tweet text, trains a TF-IDF vectorizer + LogisticRegression to classify disaster-related tweets. `explain.py` shows top TF-IDF words.
4. **Orchestrator / API** (`orchestrator/api.py`) — Flask endpoints:

   * `POST /predict/text` → classify text JSON `{"text":"..."}`
   * `POST /predict/sensor` → infer sensor window JSON `{"window":[{...}, ...]}`
5. **Fusion** (`orchestrator/orchestrator.py`) — small script that calls both endpoints and applies a simple rule (sensor anomaly OR sensor_score + ≥N disaster tweets → alert).
6. **Dashboard** (`dashboard/app_streamlit.py`) — shows sensor charts and allows classifying tweets for demo/screenshot.
7. **Colab Notebook** (`disaster_management_ai_colab.ipynb`) — all demo steps in a single notebook; recommended if you don’t want to install packages locally.

---

# What you will produce for your report (recommended)

* Classification report & confusion matrix output from `social_ml/src/train_model.py` (copy paste).
* A table comparing Threshold-only vs Sensor-ML vs Social-ML vs Fusion (use demo numbers).
* Screenshot of Streamlit dashboard (sensor chart + classification result).
* Plot of sensor time-series with shaded anomaly windows.
* Orchestrator fusion demo output JSON showing `alert: true` and reasons.

---

# How to run — Option A (EASIEST): Google Colab (no installation)

1. Open [https://colab.research.google.com](https://colab.research.google.com)
2. File → Upload notebook → select `disaster_management_ai_colab.ipynb` from this repo
3. Run cells in order. When prompted, upload:

   * `sensor_demo.csv`
   * `tweets_demo.csv`
4. At the final cell you can download `disaster_ai_outputs.zip` containing:

   * `social_model.joblib`, `sensor_model.joblib`, `sensor_window_results.csv`, `water_plot.png`, executed notebook, etc.

Use outputs/screenshots from the notebook for your PDF. This is the fastest, guaranteed-working path.

---

# How to run — Option B: Local (recommended if you need Streamlit/Flask locally)

Two ways: `venv` (native Python) or `conda` (recommended on Windows if pip build fails).

## A — Using a Python virtual environment (venv)

**1. Create & activate venv**

```bash
python -m venv venv
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
# or use cmd:
# venv\Scripts\activate.bat
# macOS / Linux
# source venv/bin/activate
```

**2. Install dependencies**

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

> If `scikit-learn` fails to build on Windows, use the Conda instructions below.

**3. Train tweet classifier**

```bash
cd social_ml/src
python train_model.py
cd ../..
```

Outputs: `social_ml/models/disaster_model.joblib` and printed classification report.

**4. Train sensor model**

```bash
python -c "from cloud_ingest.sensor_model import train_iforest; train_iforest('demo_data/sensor_demo.csv')"
```

Outputs: `models/sensor_iforest.joblib` (or `outputs` depending on code).

**5. Start Flask API**

```bash
python orchestrator/api.py
```

Server runs at `http://127.0.0.1:5000`.

**6. Start Streamlit dashboard (new terminal)**

```bash
streamlit run dashboard/app_streamlit.py
```

Open `http://localhost:8501`.

**7. Run fusion demo**

```bash
python orchestrator/orchestrator.py
```

It will call the Flask endpoints and print the fused decision.

---

## B — Using Conda (Windows recommended path)

**1. Create & activate conda env**

```bash
conda create -n dm python=3.10 -y
conda activate dm
```

**2. Install prebuilt packages**

```bash
conda install -c conda-forge scikit-learn=1.2.2 pandas=1.5.3 matplotlib joblib streamlit flask nltk emoji requests -y
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt   # if you have extra pip-only deps
```

Then follow steps 3–7 from the venv section above.

---

# API usage examples (curl)

**Text classify**

```bash
curl -X POST http://127.0.0.1:5000/predict/text -H "Content-Type: application/json" -d "{\"text\":\"Floods in my area, need help\"}"
# -> {"label":1,"confidence":0.82}
```

**Sensor window (example body: JSON list of rows)**

```bash
curl -X POST http://127.0.0.1:5000/predict/sensor -H "Content-Type: application/json" -d "{\"window\":[{\"vibration\":50,\"flame\":0,\"water\":120}, {\"vibration\":52,\"flame\":0,\"water\":140}]}"
# -> {"alert": true, "score": ...}
```

---

# Files you should commit to GitHub

* All code files (`social_ml/`, `cloud_ingest/`, `orchestrator/`, `dashboard/`)
* `disaster_management_ai_colab.ipynb` (notebook)
* `demo_data/sensor_demo.csv` and `demo_data/tweets_demo.csv` (so CI can run)
* `README.md`, `.github/workflows/run_notebook.yml`

**Do NOT commit**: `venv/`, large model binaries if >10–50MB (unless necessary). Use `.gitignore`.

---

# Colab badge — paste into README top

```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-USERNAME/YOUR-REPO/blob/main/disaster_management_ai_colab.ipynb)
```

Replace `YOUR-USERNAME/YOUR-REPO` with your GitHub username and repo name.

---

# GitHub Actions (CI) note

A workflow (`.github/workflows/run_notebook.yml`) is included to execute the Colab notebook headlessly on each `push` to `main` or on manual dispatch. The Action will upload the outputs/artifacts (executed notebook and `outputs/` files) as an artifact you can download for your report.

**Important:** for successful CI runs, ensure `demo_data/sensor_demo.csv` and `demo_data/tweets_demo.csv` are present in the repository (the notebook reads them). If the Action fails, open **Actions → run → Logs** to see the failing step and paste the log here if you want my help fixing it.

---

# Troubleshooting (most common problems)

* **scikit-learn pip build failure on Windows** → use Conda (see Conda section).
* **PowerShell activation blocked** → run:

  ```powershell
  Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
  .\venv\Scripts\Activate.ps1
  ```

  or use `cmd` activation: `venv\Scripts\activate.bat`.
* **Flask or Streamlit not connecting** → ensure you run Flask first and that both run in the same environment. Use `http://127.0.0.1:5000` in dashboard if needed.
* **CI failing because notebook cannot find CSVs** → commit `demo_data/*.csv` or modify notebook to create demo data programmatically.






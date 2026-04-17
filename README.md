# ☀️ TCU Solar Flare Forecast

A real-time solar flare prediction system built at **Texas Christian University**. It downloads HMI magnetogram images from the Helioviewer API every hour, classifies them with a custom AlexNet-based deep learning model, and displays live M/X-class flare forecasts in a Streamlit dashboard — validated against official GOES event data from LMSAL.

---

## 🗂️ Repository Structure

```
solar-flare-forecast/
│
├── app.py                     # Streamlit dashboard (main UI)
│
├── collect_data.py            # Bulk historical download: Helioviewer JP2 → JPG (Feb 2026 onward)
├── collect_latest.py          # Hourly download of the single latest HMI image
│
├── predict.py                 # Batch inference over all images in data/hmi_jpg/
├── predict_latest.py          # Incremental inference: predicts only the newest image
│
├── scrape_ssw.py              # Scrapes LMSAL/SolarSoft for GOES flare event data
├── scheduler.py               # Runs scrape_ssw.py on a 60-minute loop
│
├── eval.py                    # Evaluates TSS / HSS against M/X-class ground truth
│
├── lmsal_all_2026_clean.csv   # Ground-truth GOES flare events (auto-updated by scrape_ssw.py)
├── prediction_history.csv     # Running log of model predictions (auto-updated)
├── evaluation_results.csv     # Saved evaluation metrics
├── confusion_matrix.png       # Confusion matrix from offline evaluation
│
├── .gitignore
└── README.md
```

> **Note:** The `data/` directory (HMI images) and `new-fold1.pth` (model weights) are not tracked by git. See setup instructions below.

---

## 🔄 System Pipeline

```
[Helioviewer API]
       │
       ▼
collect_latest.py          ← runs every hour (via cron or manually)
       │  downloads latest HMI magnetogram JP2 → converts to JPG
       ▼
data/hmi_jpg/latest_jpg/
       │
       ▼
predict_latest.py          ← runs every hour
       │  loads new-fold1.pth, runs AlexNet inference
       ▼
prediction_history.csv
       │
       ▼
app.py                     ← Streamlit dashboard reads this file
       │  displays forecast, history, skill scores
       ▼
[Browser]

[LMSAL/SolarSoft]
       │
       ▼
scrape_ssw.py              ← runs every hour via scheduler.py
       │
       ▼
lmsal_all_2026_clean.csv   ← ground truth for eval.py + app.py skill scores
```

---

## 🚀 Setup

### 1. Clone the repo

```bash
git clone https://github.com/hoangthaison2306/solar-flare-forecast.git
cd solar-flare-forecast
```

### 2. Install dependencies

```bash
pip install streamlit pandas numpy torch torchvision pillow opencv-python requests beautifulsoup4 schedule
```

> OpenCV must be built with OpenJPEG support to decode `.jp2` files. The easiest install:
> ```bash
> pip install opencv-python-headless
> ```

### 3. Add the model weights

Place `new-fold1.pth` in the project root. This file is not tracked by git due to its size.

The model uses `Custom_AlexNet` from the [`explainingFullDisk`](https://github.com/SpaceML-org/ExplainingFullDiskSolarFlares) package. Install or clone it so the import resolves:

```bash
git clone https://github.com/SpaceML-org/ExplainingFullDiskSolarFlares.git explainingFullDisk
```

### 4. Download historical images (optional, for bulk evaluation)

```bash
python collect_data.py
```

Downloads hourly HMI magnetograms from Feb 1, 2026 onward and converts them to JPG under `data/hmi_jpg/`.

---

## ▶️ Running the System

### Start the LMSAL scraper (keep running in background)

```bash
python scheduler.py
```

Runs `scrape_ssw.py` immediately on startup, then every 60 minutes. Keeps `lmsal_all_2026_clean.csv` up to date.

### Fetch the latest solar image + run inference (run hourly, e.g. via cron)

```bash
python collect_latest.py
python predict_latest.py
```

Or to batch-predict all historical images at once:

```bash
python predict.py
```

### Launch the dashboard

```bash
streamlit run app.py
```

---

## 📊 Evaluation

Compute TSS and HSS against M/X-class GOES ground truth:

```bash
python eval.py
```

Example output:

```
Period           TSS      HSS
--------------------------------
Last 1 Week   +0.XXXX  +0.XXXX
Last 1 Month  +0.XXXX  +0.XXXX
Last 2 Months +0.XXXX  +0.XXXX
```

**Metric definitions (threshold: probability ≥ 0.5):**
- **TSS** (True Skill Statistic) = POD − FAR. Range [−1, 1]; 0 = no skill.
- **HSS** (Heidke Skill Score) = skill relative to random chance. Range (−∞, 1]; 1 = perfect.

A prediction is a **true positive** if any M or X-class flare starts between `image_time` and `forecast_end` (image time + 12 hours).

---

## 🛰️ Data Sources

| Source | Description |
|--------|-------------|
| [Helioviewer API](https://helioviewer.org) | HMI Line-of-Sight Magnetogram images (sourceId=19), JP2 format |
| [LMSAL / SolarSoft](https://www.lmsal.com/solarsoft/) | GOES flare event catalog, scraped hourly |

---

## 📁 What's Not in This Repo

| Path | Why excluded |
|------|--------------|
| `data/` | HMI image files — large binary data, regenerate with `collect_data.py` |
| `new-fold1.pth` | Model weights — too large for git; obtain separately |

Make sure your `.gitignore` includes:

```
data/
*.pth
*.jp2
__pycache__/
*.pyc
.env
```

---

## 🏫 Affiliation

Developed at **Texas Christian University (TCU)**.  
Model architecture based on the [ExplainingFullDisk](https://github.com/SpaceML-org/ExplainingFullDiskSolarFlares) solar flare forecasting framework.

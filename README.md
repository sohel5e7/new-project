# 🛡️ Project Panopticon: Intelligent Exam Proctoring & Behavioral Telemetry Pipeline

**EduGuard AI — Machine Learning Division | Executive Deliverable**  
**Standards Compliance:** ISO/IEC 42001 (Trustworthy AI) & Algorithmic Due Process

---

## 🏛️ Executive Summary & Scenario

With the global adoption of remote education, universities rely on **EduGuard AI** to monitor students during high-stakes examinations. The proctoring system continuously ingests multi-modal sensory data:
- **Computer Vision Gaze & Head Posture (1 Hz synchronous stream)**
- **Ambient Acoustic Telemetry (1 Hz synchronous decibel stream)**
- **Operating System Event Telemetry (asynchronous interrupts: tab switches, copy/paste, window blur)**

### The Crisis
The legacy proctoring system suffered from catastrophic algorithmic rigidity. Innocent candidates were routinely flagged for:
1. Sneezing or coughing (acoustic spikes of $75\text{ dB} - 83\text{ dB}$).
2. Glancing down at keyboards or scratch paper while writing formulas ($1-3\text{ s}$ pitch shifts).
3. Transient network packet loss ($5-15\text{ s}$ video freeze).

Universities threatened contract cancellation due to student outrage and administrative overload. This project re-architects the proctoring pipeline into an intelligent, time-series classification system that accurately detects genuine collusion while **mathematically eliminating false accusations ($\ge 90\%$ Precision target, achieved $100\%$ on test split)**.

---

## 📂 Project Architecture

```
d:/sohel/
├── data/
│   ├── video_telemetry.csv               # 1 Hz continuous telemetry (14,400 rows, 8 students)
│   └── system_events.csv                 # Asynchronous OS event logs (40 events)
├── Project_Panopticon_Executive_Notebook.ipynb  # Primary Deliverable: Full Executed Notebook
├── run_pipeline.py                       # Standalone CLI validation pipeline
├── panopticon_app.py                     # Interactive Streamlit proctoring dashboard
├── panopticon_pr_curve.png               # High-resolution Precision-Recall curve
├── requirements.txt                      # Project dependencies
└── README.md                             # Comprehensive technical documentation
```

---

## ⚡ The Four Deadly Traps & Mathematical Solutions

| # | Deadly Trap | Failure Mode of Naive System | Panopticon Mathematical Solution |
|---|---|---|---|
| **1** | **Asynchronous Merge Trap** | Inner/Outer join drops non-matching timestamps or creates thousands of disconnected null rows. | `pd.merge_asof` with `direction='nearest'` and a `2.0s` tolerance window. Timeline remains $100\%$ continuous. |
| **2** | **"Connection Dropped" Trap** | Dropping `NaN` rows destroys the continuous time index ($t=345$ jumps to $t=380$), breaking rolling windows. | Feature-flagging (`connection_lost = 1`) followed by per-student linear interpolation and `ffill`/`bfill`. |
| **3** | **Micro-Movement Noise Trap** | Instantaneous thresholding flags $1\text{ s}$ sneezes and $2\text{ s}$ keyboard glances as cheating. | Multi-horizon rolling window filters ($5\text{ s}, 10\text{ s}, 30\text{ s}$ means, std, system event density). |
| **4** | **False Accusation Trap** | Default $\theta = 0.50$ produces **38+ False Positives** under heavy class imbalance ($1.67\%$ positive rate). | Posterior probability estimation via `.predict_proba()` and Precision-Recall optimization to select $\theta^* = 0.75$, yielding $\mathbf{0}$ False Positives and $100.0\%$ Precision. |

---

## 📊 Quantitative Benchmark Results

Evaluated across **3,600 held-out candidate exam seconds (1 hour of testing)**:

| Metric | Legacy Naive Model ($\theta = 0.50$) | Panopticon Default ($\theta = 0.50$) | Panopticon Ethical ($\theta^* = 0.75$) |
|---|:---:|:---:|:---:|
| **False Positives (Innocents Accused)** | **38** | **0** | **0 (100% Elimination)** |
| **Precision (PPV)** | $61.2\%$ | $100.0\%$ | **$100.0\%$** |
| **Recall (Sensitivity)** | $98.3\%$ | $91.7\%$ | **$90.0\%$** |
| **Average Precision (PR-AUC)** | $0.684$ | $0.985$ | **$0.985$** |
| **Innocent Sneeze ($82.2\text{ dB}$)** | Flagged as Cheater ❌ | Cleared ✅ | **Cleared ✅ ($P=1.7\%$)** |
| **Keyboard Glance ($-27.7^\circ$)** | Flagged as Cheater ❌ | Cleared ✅ | **Cleared ✅ ($P=1.7\%$)** |
| **True Sustained Cheating** | Flagged ✅ | Flagged ✅ | **Flagged ✅ ($P=97.3\%$)** |

---

## 🚀 Execution & Verification Guide

### 1. Execute the Jupyter Notebook
Open and run `Project_Panopticon_Executive_Notebook.ipynb` in VS Code or JupyterLab:
```bash
jupyter notebook Project_Panopticon_Executive_Notebook.ipynb
```
*The notebook is already fully executed with rich plots, interactive heatmaps, and classification matrices.*

### 2. Run the Command-Line Pipeline
Execute the automated validation suite from terminal:
```powershell
python run_pipeline.py
```

### 3. Launch the Interactive Dashboard
Launch the real-time proctoring Streamlit app:
```powershell
streamlit run panopticon_app.py
```
Explore live replay of candidate sessions, toggle between legacy and ethical decision thresholds, and inspect corner-case audits.

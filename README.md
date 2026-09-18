# 🛡️ Edufi Universal Machine Learning & Cybersecurity Portal

An enterprise-grade, multi-role web platform integrating interactive machine learning workspaces, real-time data science sandboxes, and an ensemble network anomaly & DDoS intrusion detection engine with built-in privacy protection and role-based access control (RBAC).

---

## 🌟 Key Features

### 🔐 Multi-Tier Role-Based Authentication
- **⚡ Admin (`admin / admin123`)**: Full system access to all 5 AI/ML classrooms, model retraining metrics, unsupervised visualizations, and the DDoS Threat Center.
- **🔬 Analyst (`analyst / data123`)**: Dedicated dataset processing workspace with anomaly detection scan capabilities.
- **👁️ Viewer (`viewer / view123`)**: Read-only oversight dashboard where sensitive business metrics and network parameters are masked/blurred via privacy firewall filters.

---

### 🧪 Interactive Machine Learning Workspaces

1. **🩺 Task 1 — Diabetes Risk Predictor**
   - Supervised classification using Random Forest, Decision Trees, and Logistic Regression.
   - Interactive SVG risk gauges and dynamic probability confidence scores.

2. **📈 Task 2 — Sales & Advertising Linear Regression**
   - Live budget estimation against multi-channel ad spend (TV, Radio, Newspaper).
   - Real-time Chart.js interactive scatter plots and regression line fitting.

3. **🧹 Task 3 — Titanic Data Preprocessing Playground**
   - Step-by-step feature engineering: Missing value imputation, categorical encoding, feature extraction, and feature scaling.

4. **⚖️ Task 4 — Supervised Survival Comparator**
   - Side-by-side performance evaluation between Decision Tree Classifiers and Linear Probability Models.
   - Detailed confusion matrices, precision/recall metrics, and feature importance rankings.

5. **🌌 Task 5 — Unsupervised PCA & K-Means Clustering**
   - 2D Principal Component Analysis (PCA) projection.
   - Dynamic K-Means clustering with configurable cluster counts, centroid tracking, and demographic breakdowns.

6. **🚨 Major Project — DDoS Anomaly Detection & Threat Center**
   - Ensemble classification engine pairing **Random Forest** and multi-threaded chunked **Support Vector Machines (SVM)**.
   - Analyzes 50+ network flow features from PCAP traffic logs.
   - Generates automated intrusion distribution pie charts, threat count bar graphs, and downloadable segmented captures (`ddos_predictions.csv` and `benign_predictions.csv`).

---

## 🛠️ Technology Stack

- **Backend**: Python 3.13, Flask, Scikit-Learn, NumPy, Pandas, Joblib
- **Frontend**: HTML5, Modern CSS3 Glassmorphism, Bootstrap, Chart.js, Vanilla JS
- **Machine Learning**: SVM, Random Forest, Decision Trees, Linear Regression, Logistic Regression, PCA, K-Means
- **Data Visualization**: Matplotlib (Agg Engine), Chart.js
- **Security & Firewall**: Role-Based Session Guards, Feature Redaction, Automated CSV sanitization

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ashwinkumar07/Multi-Agent-.git
   cd Multi-Agent-
   ```

2. **Install required dependencies:**
   ```bash
   pip install flask pandas numpy scikit-learn matplotlib joblib
   ```

3. **Launch the Portal:**
   - **On Windows (PowerShell / Terminal):**
     ```powershell
     python app.py
     ```
   - **Or double-click `RUN_PORTAL.bat`** in the project folder.

4. **Access in your browser:**
   ```
   http://127.0.0.1:8080
   ```

---

## 🔑 Login Credentials

| Role | Username | Password | Access Level |
| :--- | :--- | :--- | :--- |
| **⚡ Admin** | `admin` | `admin123` | Full Hub Portal & All Workspaces |
| **🔬 Analyst** | `analyst` | `data123` | Dataset Upload & DDoS Scanner |
| **👁️ Viewer** | `viewer` | `view123` | Read-Only Dashboard (Redacted Content) |

---

## 📁 Repository Structure

```
├── app.py                     # Main Flask application & routing with RBAC guards
├── ml_engine.py               # Machine learning pipelines, training & prediction logic
├── data_loader.py             # Dataset loader and caching utility
├── RUN_PORTAL.bat             # 1-Click launcher script
├── datasets/                  # Benchmark datasets (Diabetes, Titanic, Advertising)
├── model/                     # Serialized pre-trained model artifacts (.pkl)
├── static/
│   ├── css/style.css          # Glassmorphism dark/modern theme styling
│   ├── js/main.js             # UI interactions & slider bindings
│   └── vendor/                # Local vendor libraries (Bootstrap, Chart.js, jQuery)
├── templates/                 # Jinja2 template views (Admin, Analyst, Viewer, Labs)
└── Task_files/                # Major Project DDoS models, logs, and artifacts
```

---

## 👤 Author
- **Ashwin Kumar** — [@Ashwinkumar07](https://github.com/Ashwinkumar07)

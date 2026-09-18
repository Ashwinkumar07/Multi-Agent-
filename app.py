import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, render_template, jsonify, send_from_directory, flash, redirect, url_for, session
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

import data_loader
import ml_engine

# Absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
MAJOR_PROJECT_DIR = os.path.join(BASE_DIR, 'Task_files', 'major project', 'anomaly_detection')

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.secret_key = 'edufi_ml_portal_secret_key'

# Load the trained DDoS models from major project folder
DDOS_MODEL_DIR = os.path.join(MAJOR_PROJECT_DIR, 'model')
svm_model = joblib.load(os.path.join(DDOS_MODEL_DIR, 'svm_model.pkl'))
rf_model = joblib.load(os.path.join(DDOS_MODEL_DIR, 'random_forest_model.pkl'))

# DDoS Feature columns
features = [
    ' Source Port', ' Destination Port', ' Protocol', ' Flow Duration',
    ' Total Fwd Packets', ' Total Backward Packets',
    'Total Length of Fwd Packets', ' Total Length of Bwd Packets',
    ' Fwd Packet Length Max', ' Fwd Packet Length Min',
    ' Fwd Packet Length Mean', ' Fwd Packet Length Std',
    'Bwd Packet Length Max', ' Bwd Packet Length Min',
    ' Bwd Packet Length Mean', ' Bwd Packet Length Std',
    ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min', 'Fwd IAT Total',
    ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max', ' Fwd IAT Min',
    'Bwd IAT Total', ' Bwd IAT Mean', ' Bwd IAT Std', ' Bwd IAT Max',
    ' Bwd IAT Min', 'Fwd PSH Flags', ' Bwd PSH Flags',
    ' Fwd URG Flags', ' Bwd URG Flags', ' Fwd Header Length', ' Bwd Header Length',
    'Fwd Packets/s', ' Bwd Packets/s', ' Min Packet Length',
    ' Max Packet Length', ' Packet Length Mean', ' Packet Length Std',
    ' Packet Length Variance', 'FIN Flag Count', ' SYN Flag Count',
    ' RST Flag Count', ' PSH Flag Count', ' ACK Flag Count',
    ' URG Flag Count', ' CWE Flag Count', ' ECE Flag Count'
]

NUM_THREADS = 10

def predict_svm_in_chunks(X_scaled):
    chunks = np.array_split(X_scaled, 20)
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        results = list(executor.map(svm_model.predict, chunks))
    return np.concatenate(results)

# Initialize datasets, train models, and enforce login authentication
@app.before_request
def before_request_handler():
    # Only run once on first request to initialize and train ML models
    if not hasattr(app, '_initialized'):
        print("[Portal] Initializing and preparing all datasets...")
        data_loader.prepare_all_datasets()
        print("[Portal] Training and caching models...")
        ml_engine.train_and_cache_everything()
        app._initialized = True
        
    # Enforce basic authentication
    allowed_routes = ['login', 'static']
    endpoint = request.endpoint
    if endpoint and endpoint not in allowed_routes and not request.path.startswith('/static'):
        if not session.get('logged_in'):
            return redirect(url_for('login'))

# =====================================================================
# ROUTING & WEB APPS
# =====================================================================

# Role-based credentials
USERS = {
    'admin':    {'password': 'admin123',  'role': 'admin'},
    'analyst':  {'password': 'data123',   'role': 'analyst'},
    'viewer':   {'password': 'view123',   'role': 'viewer'},
}

def get_role():
    """Return current user's role from session."""
    return session.get('role')

def require_role(*roles):
    """Check if logged-in user has one of the allowed roles."""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    if get_role() not in roles:
        flash(f'Access denied. This section requires: {", ".join(roles)} role.', 'error')
        return redirect(url_for('role_home'))
    return None

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Sign In Page with role-based authentication."""
    if session.get('logged_in'):
        return redirect(url_for('role_home'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        user = USERS.get(username)
        if user and user['password'] == password:
            session['logged_in'] = True
            session['username'] = username
            session['role'] = user['role']
            session.permanent = True
            return redirect(url_for('role_home'))
        else:
            flash('Invalid username or password.', 'error')

    return render_template('login.html')

@app.route('/role_home')
def role_home():
    """Redirect user to their role-appropriate landing page."""
    role = get_role()
    if role == 'admin':
        return redirect(url_for('dashboard'))
    elif role == 'analyst':
        return redirect(url_for('analyst_upload'))
    elif role == 'viewer':
        return redirect(url_for('viewer_dashboard'))
    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    """Clear session data and redirect to login."""
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
def dashboard():
    """Universal ML Portal Homepage — Admin only."""
    guard = require_role('admin')
    if guard: return guard
    return render_template('index.html', active_page='dashboard',
                           username=session.get('username'), role=get_role())

# ── ANALYST ROLE ─────────────────────────────────────────────────────────────
@app.route('/analyst')
def analyst_upload():
    """Analyst-only upload workspace."""
    guard = require_role('analyst', 'admin')
    if guard: return guard
    return render_template('analyst.html', active_page='analyst',
                           username=session.get('username'), role=get_role())

# ── VIEWER ROLE ───────────────────────────────────────────────────────────────
@app.route('/viewer')
def viewer_dashboard():
    """Viewer-only read-only masked dashboard."""
    guard = require_role('viewer', 'admin')
    if guard: return guard
    return render_template('viewer.html', active_page='viewer',
                           username=session.get('username'), role=get_role())

# Task 1: Diabetes Prediction
@app.route('/diabetes')
def diabetes_dashboard():
    """Interactive diabetes predictor workspace — Admin only."""
    guard = require_role('admin')
    if guard: return guard
    metrics = ml_engine.train_diabetes_models()
    return render_template('diabetes.html', active_page='diabetes', metrics=metrics,
                           username=session.get('username'), role=get_role())

@app.route('/api/predict_diabetes', methods=['POST'])
def api_predict_diabetes():
    """Real-time diabetes prediction endpoint."""
    try:
        inputs = [
            float(request.form.get('pregnancies', 3)),
            float(request.form.get('glucose', 115)),
            float(request.form.get('blood_pressure', 72)),
            float(request.form.get('skin_thickness', 23)),
            float(request.form.get('insulin', 79)),
            float(request.form.get('bmi', 32.0)),
            float(request.form.get('dpf', 0.47)),
            float(request.form.get('age', 33))
        ]
        model_name = request.form.get('model_name', 'random_forest')
        result = ml_engine.predict_diabetes(inputs, model_name)
        result['success'] = True
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Task 2: Simple Linear Regression
@app.route('/linear_regression')
def regression_dashboard():
    """Simple linear regression analyzer workspace — Admin only."""
    guard = require_role('admin')
    if guard: return guard
    metrics = ml_engine.train_sales_regression()
    return render_template('linear_regression.html', active_page='regression', metrics=metrics,
                           username=session.get('username'), role=get_role())

@app.route('/api/sales_regression_data')
def api_sales_regression_data():
    """Return sales scatter plot and line fit data."""
    data = ml_engine.train_sales_regression()
    return jsonify(data)

# Task 3: Titanic Preprocessing
@app.route('/titanic_preprocessing')
def preprocessing_dashboard():
    """Titanic step-by-step cleaning playground — Admin only."""
    guard = require_role('admin')
    if guard: return guard
    stages = ml_engine.get_titanic_preprocessing_stages()
    return render_template('titanic_preprocessing.html', active_page='preprocessing', stages=stages,
                           username=session.get('username'), role=get_role())

# Task 4: Titanic Supervised Comparison
@app.route('/titanic_supervised')
def titanic_supervised_dashboard():
    """Titanic supervised classification comparator — Admin only."""
    guard = require_role('admin')
    if guard: return guard
    metrics = ml_engine.train_titanic_supervised()
    return render_template('titanic_supervised.html', active_page='supervised', metrics=metrics,
                           username=session.get('username'), role=get_role())

# Task 5: Titanic Unsupervised PCA & K-Means
@app.route('/titanic_unsupervised')
def titanic_unsupervised_dashboard():
    """Titanic unsupervised PCA scatter & clustering playground — Admin only."""
    guard = require_role('admin')
    if guard: return guard
    data = ml_engine.get_titanic_unsupervised(n_clusters=3)
    return render_template('titanic_unsupervised.html', active_page='unsupervised', variance=data['explained_variance'],
                           username=session.get('username'), role=get_role())

@app.route('/api/titanic_unsupervised_data')
def api_titanic_unsupervised_data():
    """Return dynamically calculated K-Means clusters and demographics."""
    k = int(request.args.get('k', 3))
    data = ml_engine.get_titanic_unsupervised(n_clusters=k)
    return jsonify(data)

# Major Project: DDoS Anomaly Detection
@app.route('/anomaly_detection', methods=['GET', 'POST'])
def anomaly_detection():
    """DDoS anomaly detection dashboard — Admin + Analyst."""
    guard = require_role('admin', 'analyst')
    if guard: return guard
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
            
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
            
        if file and file.filename.endswith('.csv'):
            try:
                # Read uploaded CSV
                df = pd.read_csv(file)
                
                # Check for required features
                missing_cols = [c for c in features if c not in df.columns]
                if missing_cols:
                    flash(f"CSV file is missing {len(missing_cols)} standard network flow features. Please upload a standard PCAP flow dataset.", 'error')
                    return redirect(request.url)
                
                # Extract and scale features
                X = df[features]
                scaler = MinMaxScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Predict
                rf_pred = rf_model.predict(X_scaled)
                svm_pred = predict_svm_in_chunks(X_scaled)
                
                combined_pred = (rf_pred + svm_pred) >= 1
                combined_pred_str = np.where(combined_pred, 'DDoS', 'BENIGN')
                
                # Save predictions back to dataframe
                df['Predicted_Label'] = combined_pred_str
                
                # Generate distribution pie chart
                labels = ['BENIGN', 'DDoS']
                sizes = [np.sum(combined_pred_str == 'BENIGN'), np.sum(combined_pred_str == 'DDoS')]
                
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['#10b981', '#ef4444'], startangle=90)
                ax.set_title('Packet Intrusion Ratios')
                pie_chart_path = os.path.join(STATIC_DIR, 'prediction_pie_chart.png')
                fig.savefig(pie_chart_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                
                # Generate bar chart
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(labels, sizes, color=['#10b981', '#ef4444'])
                ax.set_xlabel('Classification')
                ax.set_ylabel('Total Packets')
                ax.set_title('Intrusion Count Breakdown')
                bar_chart_path = os.path.join(STATIC_DIR, 'prediction_bar_chart.png')
                fig.savefig(bar_chart_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                
                # Segment predictions for download
                ddos_df = df[df['Predicted_Label'] == 'DDoS']
                benign_df = df[df['Predicted_Label'] == 'BENIGN']
                
                ddos_csv = 'ddos_predictions.csv'
                benign_csv = 'benign_predictions.csv'
                
                ddos_df.to_csv(os.path.join(STATIC_DIR, ddos_csv), index=False)
                benign_df.to_csv(os.path.join(STATIC_DIR, benign_csv), index=False)
                
                # Evaluate accuracy if true label column is present
                accuracy_text = "Evaluation Bypassed"
                report_text = "Evaluation requires ground-truth labels. Since you uploaded a raw network log, accuracy metrics are bypassed but all DDoS flows have been successfully isolated."
                
                label_col = [c for c in df.columns if c.strip().lower() == 'label']
                if label_col:
                    y_true = df[label_col[0]].astype(str).str.strip()
                    # Standardize labels
                    y_true = y_true.map({'BENIGN': 'BENIGN', 'DDoS': 'DDoS', '0': 'BENIGN', '1': 'DDoS'})
                    
                    if not y_true.isnull().all():
                        acc = accuracy_score(y_true, combined_pred_str)
                        report = classification_report(y_true, combined_pred_str)
                        accuracy_text = f"{acc * 100:.4f}% Accuracy"
                        report_text = report
                
                return render_template(
                    'results.html',
                    active_page='anomaly',
                    accuracy=accuracy_text,
                    report=report_text,
                    pie_chart='prediction_pie_chart.png',
                    bar_chart='prediction_bar_chart.png',
                    ddos_csv=ddos_csv,
                    benign_csv=benign_csv
                )
                
            except Exception as e:
                flash(f"Error processing scan: {str(e)}", 'error')
                return redirect(request.url)
        else:
            flash('Invalid file format. Please upload a .csv file.', 'error')
            return redirect(request.url)
            
    return render_template('ddos.html', active_page='anomaly')

# Download Route
@app.route('/download/<filename>')
def download_file(filename):
    """Download reports or test logs directly."""
    # If the user requests test.csv, download from major project folder
    if filename == 'test.csv':
        return send_from_directory(MAJOR_PROJECT_DIR, 'test.csv')
    return send_from_directory(STATIC_DIR, filename)

if __name__ == '__main__':
    # Ensure static directory exists
    if not os.path.exists(STATIC_DIR):
        os.makedirs(STATIC_DIR)
        
    # Check if we can run as a standalone desktop app window using pywebview
    try:
        import webview
        from threading import Thread
        
        # Start Flask in a background daemon thread
        # We turn off debug mode and reloader because debug reloader doesn't play well with GUI loops
        def run_flask():
            app.run(host='127.0.0.1', port=8080, debug=False, threaded=True)
            
        flask_thread = Thread(target=run_flask)
        flask_thread.daemon = True
        flask_thread.start()
        
        print("[Portal] Started Flask engine in background thread. Launching native desktop viewport...")
        
        # Create a dedicated webview window
        webview.create_window(
            title="Edufi Universal AI/ML & Cybersecurity Portal",
            url="http://127.0.0.1:8080",
            width=1280,
            height=850,
            resizable=True,
            min_size=(1000, 700)
        )
        
        # Start the GUI window loop
        webview.start()
        
    except ImportError:
        # Fallback to standard browser opening mode if pywebview is missing
        import webbrowser
        from threading import Timer
        
        def open_browser():
            webbrowser.open_new("http://127.0.0.1:8080")
            
        if not os.environ.get("WERKZEUG_RUN_MAIN"):
            Timer(1.2, open_browser).start()
            
        app.run(debug=True, port=8080)

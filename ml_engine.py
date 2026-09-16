import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, 'datasets')
MODEL_DIR = os.path.join(BASE_DIR, 'model')

# =====================================================================
# TASK 1: DIABETES PREDICTION
# =====================================================================

def train_diabetes_models():
    """Train and cache Logistic Regression, Decision Tree, and Random Forest on Diabetes dataset."""
    filepath = os.path.join(DATASETS_DIR, 'diabetes.csv')
    df = pd.read_csv(filepath)
    
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'diabetes_scaler.pkl'))
    
    models = {
        'logistic_regression': LogisticRegression(random_state=42),
        'decision_tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Save model
        joblib.dump(model, os.path.join(MODEL_DIR, f'diabetes_{name}.pkl'))
        
        # Evaluate
        cm = confusion_matrix(y_test, y_pred)
        results[name] = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'f1': float(f1_score(y_test, y_pred)),
            'confusion_matrix': cm.tolist()
        }
        
    print("Diabetes models trained and saved successfully.")
    return results

def predict_diabetes(inputs, model_name='random_forest'):
    """Make real-time prediction using standard diabetes inputs."""
    scaler = joblib.load(os.path.join(MODEL_DIR, 'diabetes_scaler.pkl'))
    model = joblib.load(os.path.join(MODEL_DIR, f'diabetes_{model_name}.pkl'))
    
    # Inputs list: [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
    inputs_scaled = scaler.transform([inputs])
    
    pred = int(model.predict(inputs_scaled)[0])
    prob = model.predict_proba(inputs_scaled)[0]
    
    return {
        'prediction': pred,
        'probability_no': float(prob[0]),
        'probability_yes': float(prob[1])
    }

# =====================================================================
# TASK 2: SIMPLE LINEAR REGRESSION
# =====================================================================

def train_sales_regression():
    """Train Simple Linear Regression model for Sales & Advertising."""
    filepath = os.path.join(DATASETS_DIR, 'sales_advertising.csv')
    df = pd.read_csv(filepath)
    
    X = df[['Advertising']]
    y = df['Sales']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    joblib.dump(model, os.path.join(MODEL_DIR, 'sales_reg.pkl'))
    
    y_pred = model.predict(X_test)
    
    # Model params
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    mse = float(mean_squared_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    
    # Prepare data points for visualization
    points = df.to_dict(orient='records')
    
    # Sort for fitted line plotting
    x_line = np.linspace(df['Advertising'].min(), df['Advertising'].max(), 50)
    y_line = model.predict(x_line.reshape(-1, 1))
    line_data = [{'x': float(x), 'y': float(y)} for x, y in zip(x_line, y_line)]
    
    print("Sales regression model trained and saved successfully.")
    return {
        'slope': slope,
        'intercept': intercept,
        'mse': mse,
        'r2': r2,
        'points': points,
        'line_data': line_data
    }

def predict_sales(advertising_spend):
    """Predict sales based on advertising spend."""
    model = joblib.load(os.path.join(MODEL_DIR, 'sales_reg.pkl'))
    pred = float(model.predict([[advertising_spend]])[0])
    return {
        'advertising': advertising_spend,
        'predicted_sales': pred
    }

# =====================================================================
# TASK 3, 4, 5: TITANIC DATA PIPELINE (CLEANING, SUPERVISED, UNSUPERVISED)
# =====================================================================

def get_titanic_preprocessing_stages():
    """Return previews of Titanic dataset at various preprocessing stages."""
    filepath = os.path.join(DATASETS_DIR, 'titanic.csv')
    df_raw = pd.read_csv(filepath)
    
    stages = {}
    stages['stage0_raw'] = {
        'columns': df_raw.columns.tolist(),
        'sample': df_raw.head(10).replace({np.nan: None}).to_dict(orient='records'),
        'missing': df_raw.isnull().sum().to_dict(),
        'shape': df_raw.shape
    }
    
    # Stage 1: Handle Missing Values
    df_cleaned = df_raw.copy()
    df_cleaned['Age'] = df_cleaned['Age'].fillna(df_cleaned['Age'].median())
    df_cleaned['Embarked'] = df_cleaned['Embarked'].fillna(df_cleaned['Embarked'].mode()[0])
    df_cleaned = df_cleaned.drop('Cabin', axis=1) # Drop Cabin due to high missing values (>70%)
    
    stages['stage1_cleaned'] = {
        'columns': df_cleaned.columns.tolist(),
        'sample': df_cleaned.head(10).replace({np.nan: None}).to_dict(orient='records'),
        'missing': df_cleaned.isnull().sum().to_dict(),
        'shape': df_cleaned.shape
    }
    
    # Stage 2: Feature Engineering
    df_engineered = df_cleaned.copy()
    # Extract Title
    df_engineered['Title'] = df_engineered['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    # Group rare titles
    rare_titles = ['Dr', 'Rev', 'Col', 'Major', 'Mlle', 'Countess', 'Ms', 'Lady', 'Jonkheer', 'Don', 'Mme', 'Capt', 'Sir']
    df_engineered['Title'] = df_engineered['Title'].replace(rare_titles, 'Rare')
    df_engineered['Title'] = df_engineered['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
    # Family Size
    df_engineered['FamilySize'] = df_engineered['SibSp'] + df_engineered['Parch'] + 1
    # Age Categories
    df_engineered['AgeCategory'] = pd.cut(df_engineered['Age'], bins=[0, 12, 18, 35, 60, 100], labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'Senior']).astype(str)
    
    stages['stage2_engineered'] = {
        'columns': df_engineered.columns.tolist(),
        'sample': df_engineered.head(10).replace({np.nan: None}).to_dict(orient='records'),
        'missing': df_engineered.isnull().sum().to_dict(),
        'shape': df_engineered.shape
    }
    
    # Stage 3: Categorical Encoding
    df_encoded = df_engineered.copy()
    le_sex = LabelEncoder()
    df_encoded['Sex'] = le_sex.fit_transform(df_encoded['Sex'])
    
    le_embarked = LabelEncoder()
    df_encoded['Embarked'] = le_embarked.fit_transform(df_encoded['Embarked'])
    
    le_title = LabelEncoder()
    df_encoded['Title'] = le_title.fit_transform(df_encoded['Title'])
    
    le_agecat = LabelEncoder()
    df_encoded['AgeCategory'] = le_agecat.fit_transform(df_encoded['AgeCategory'])
    
    # Drop non-numerical columns for modeling
    cols_to_drop = ['PassengerId', 'Name', 'Ticket']
    df_encoded = df_encoded.drop(cols_to_drop, axis=1)
    
    stages['stage3_encoded'] = {
        'columns': df_encoded.columns.tolist(),
        'sample': df_encoded.head(10).replace({np.nan: None}).to_dict(orient='records'),
        'missing': df_encoded.isnull().sum().to_dict(),
        'shape': df_encoded.shape
    }
    
    # Stage 4: Feature Scaling
    scaler = MinMaxScaler()
    scaled_features = df_encoded.drop('Survived', axis=1).columns.tolist()
    df_scaled = df_encoded.copy()
    df_scaled[scaled_features] = scaler.fit_transform(df_scaled[scaled_features])
    
    stages['stage4_scaled'] = {
        'columns': df_scaled.columns.tolist(),
        'sample': df_scaled.head(10).replace({np.nan: None}).to_dict(orient='records'),
        'missing': df_scaled.isnull().sum().to_dict(),
        'shape': df_scaled.shape
    }
    
    return stages

def train_titanic_supervised():
    """Train Decision Tree and Linear Probability Model (Linear Regression) on Titanic survival."""
    filepath = os.path.join(DATASETS_DIR, 'titanic.csv')
    df = pd.read_csv(filepath)
    
    # Apply standard pipeline
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Feature engineering
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    rare_titles = ['Dr', 'Rev', 'Col', 'Major', 'Mlle', 'Countess', 'Ms', 'Lady', 'Jonkheer', 'Don', 'Mme', 'Capt', 'Sir']
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
    df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['AgeCategory'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'Senior']).astype(str)
    
    # Encodings
    for col in ['Sex', 'Embarked', 'Title', 'AgeCategory']:
        df[col] = LabelEncoder().fit_transform(df[col])
        
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, errors='ignore')
    
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Decision Tree Classifier
    dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt_model.fit(X_train_scaled, y_train)
    dt_pred = dt_model.predict(X_test_scaled)
    
    # 2. Linear Regression (Linear Probability Model)
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    lr_pred_raw = lr_model.predict(X_test_scaled)
    lr_pred = (lr_pred_raw >= 0.5).astype(int) # Threshold at 0.5
    
    # Evaluation Metrics
    # Decision Tree
    dt_cm = confusion_matrix(y_test, dt_pred).tolist()
    dt_metrics = {
        'accuracy': float(accuracy_score(y_test, dt_pred)),
        'precision': float(precision_score(y_test, dt_pred)),
        'recall': float(recall_score(y_test, dt_pred)),
        'f1': float(f1_score(y_test, dt_pred)),
        'mse': float(mean_squared_error(y_test, dt_pred)),
        'r2': float(r2_score(y_test, dt_pred)),
        'confusion_matrix': dt_cm
    }
    
    # Linear Probability Model (Linear Regression)
    lr_cm = confusion_matrix(y_test, lr_pred).tolist()
    lr_metrics = {
        'accuracy': float(accuracy_score(y_test, lr_pred)),
        'precision': float(precision_score(y_test, lr_pred)),
        'recall': float(recall_score(y_test, lr_pred)),
        'f1': float(f1_score(y_test, lr_pred)),
        'mse': float(mean_squared_error(y_test, lr_pred_raw)), # MSE on continuous probabilities
        'r2': float(r2_score(y_test, lr_pred_raw)), # R2 on continuous probabilities
        'confusion_matrix': lr_cm
    }
    
    # Feature Importances (from Decision Tree)
    importances = dt_model.feature_importances_
    features_list = X.columns.tolist()
    feature_importance = [{'feature': f, 'importance': float(i)} for f, i in zip(features_list, importances)]
    feature_importance = sorted(feature_importance, key=lambda x: x['importance'], reverse=True)
    
    print("Titanic supervised models evaluated successfully.")
    return {
        'decision_tree': dt_metrics,
        'linear_regression': lr_metrics,
        'feature_importance': feature_importance
    }

def get_titanic_unsupervised(n_clusters=3):
    """Perform PCA and K-Means clustering on Titanic, returning coordinates and profiles."""
    filepath = os.path.join(DATASETS_DIR, 'titanic.csv')
    df = pd.read_csv(filepath)
    
    # Cache survival labels for later overlay
    y_survived = df['Survived'].tolist()
    pclass_list = df['Pclass'].tolist()
    gender_list = df['Sex'].tolist()
    
    # Basic Preprocessing
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    rare_titles = ['Dr', 'Rev', 'Col', 'Major', 'Mlle', 'Countess', 'Ms', 'Lady', 'Jonkheer', 'Don', 'Mme', 'Capt', 'Sir']
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
    df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['AgeCategory'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'Senior']).astype(str)
    
    # Keep standard numerical and encoded features for clustering
    df_cluster = df.copy()
    for col in ['Sex', 'Embarked', 'Title', 'AgeCategory']:
        df_cluster[col] = LabelEncoder().fit_transform(df_cluster[col])
        
    features_to_use = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'Title', 'AgeCategory']
    X = df_cluster[features_to_use]
    
    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Principal Component Analysis (reduce to 2 dimensions for plotting)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # K-Means Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_transform(X_scaled) # Run K-Means on scaled original features
    labels = kmeans.labels_.tolist()
    
    # Prepare coordinates for Chart.js 2D Scatter plot
    scatter_data = []
    for i in range(len(X_pca)):
        scatter_data.append({
            'x': float(X_pca[i, 0]),
            'y': float(X_pca[i, 1]),
            'cluster': int(labels[i]),
            'survived': int(y_survived[i]),
            'pclass': int(pclass_list[i]),
            'sex': gender_list[i],
            'age': float(df['Age'].iloc[i]),
            'fare': float(df['Fare'].iloc[i])
        })
        
    # Analyze Cluster Profiles (demographics per cluster)
    df['Cluster'] = labels
    profiles = []
    for c in range(n_clusters):
        cluster_subset = df[df['Cluster'] == c]
        profiles.append({
            'cluster_id': c,
            'count': int(len(cluster_subset)),
            'avg_age': float(cluster_subset['Age'].mean()),
            'avg_fare': float(cluster_subset['Fare'].mean()),
            'survival_rate': float(cluster_subset['Survived'].mean() * 100),
            'female_ratio': float((cluster_subset['Sex'] == 'female').mean() * 100),
            'first_class_ratio': float((cluster_subset['Pclass'] == 1).mean() * 100)
        })
        
    # Compute explained variance
    variance_ratio = pca.explained_variance_ratio_.tolist()
    
    print(f"Titanic unsupervised (K={n_clusters}) completed successfully.")
    return {
        'scatter_data': scatter_data,
        'profiles': profiles,
        'explained_variance': variance_ratio
    }

def train_and_cache_everything():
    """Run model training for all tasks on initialization."""
    print("Training and caching all dashboard models...")
    train_diabetes_models()
    train_sales_regression()
    train_titanic_supervised()
    print("All backend models have been successfully trained and cached.")

if __name__ == '__main__':
    train_and_cache_everything()

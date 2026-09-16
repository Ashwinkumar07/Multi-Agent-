import os
import pandas as pd
import numpy as np
import urllib.request

# Define absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, 'datasets')
MODEL_DIR = os.path.join(BASE_DIR, 'model')

def initialize_directories():
    """Create necessary directories if they don't exist."""
    for directory in [DATASETS_DIR, MODEL_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def download_titanic_dataset():
    """Download standard Titanic dataset if not cached."""
    filepath = os.path.join(DATASETS_DIR, 'titanic.csv')
    if not os.path.exists(filepath):
        print("Downloading Titanic dataset...")
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        try:
            urllib.request.urlretrieve(url, filepath)
            print("Titanic dataset downloaded successfully.")
        except Exception as e:
            print(f"Error downloading Titanic dataset: {e}")
            # Fallback to creating a miniature mock dataset to avoid crash
            create_mock_titanic(filepath)
    else:
        print("Titanic dataset already cached.")
    return filepath

def download_diabetes_dataset():
    """Download Pima Indians Diabetes dataset if not cached."""
    filepath = os.path.join(DATASETS_DIR, 'diabetes.csv')
    if not os.path.exists(filepath):
        print("Downloading Diabetes dataset...")
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        try:
            urllib.request.urlretrieve(url, filepath)
            # Add standard headers
            headers = [
                'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
            ]
            df = pd.read_csv(filepath, names=headers)
            df.to_csv(filepath, index=False)
            print("Diabetes dataset downloaded and formatted successfully.")
        except Exception as e:
            print(f"Error downloading Diabetes dataset: {e}")
            create_mock_diabetes(filepath)
    else:
        print("Diabetes dataset already cached.")
    return filepath

def generate_sales_advertising_dataset():
    """Generate or retrieve synthetic Sales and Advertising dataset."""
    filepath = os.path.join(DATASETS_DIR, 'sales_advertising.csv')
    if not os.path.exists(filepath):
        print("Generating Sales & Advertising dataset...")
        np.random.seed(42)
        n_samples = 100
        # Advertising budget (in thousands)
        advertising = np.round(np.random.uniform(10.0, 150.0, n_samples), 2)
        # Sales (in thousands) with positive correlation and random noise
        sales = np.round(8.5 + 0.28 * advertising + np.random.normal(0, 4.5, n_samples), 2)
        # Ensure sales are not negative
        sales = np.clip(sales, 1.0, None)
        
        df = pd.DataFrame({
            'Advertising': advertising,
            'Sales': sales
        })
        df.to_csv(filepath, index=False)
        print("Sales & Advertising dataset generated successfully.")
    else:
        print("Sales & Advertising dataset already exists.")
    return filepath

def create_mock_titanic(filepath):
    """Create a miniature fallback mock Titanic dataset."""
    print("Creating mock Titanic dataset...")
    data = {
        'PassengerId': list(range(1, 21)),
        'Survived': [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1],
        'Pclass': [3, 1, 3, 1, 3, 3, 1, 3, 3, 2, 3, 1, 3, 3, 3, 2, 3, 3, 3, 3],
        'Name': [
            'Braund, Mr. Owen Harris', 'Cumings, Mrs. John Bradley (Florence Briggs Thayer)',
            'Heikkinen, Miss. Laina', 'Futrelle, Mrs. Jacques Heath (Lily May Peel)',
            'Allen, Mr. William Henry', 'Moran, Mr. James', 'McCarthy, Mr. Timothy J',
            'Palsson, Master. Gosta Leonard', 'Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)',
            'Nasser, Mrs. Nicholas (Adele Achem)', 'Sandstrom, Miss. Marguerite Rut',
            'Bonnell, Miss. Elizabeth', 'Saundercock, Mr. William Henry', 'Andersson, Mr. Anders Johan',
            'Vestrom, Miss. Hulda Amanda Adolfina', 'Hewlett, Mrs. (Mary D Kingcome) ',
            'Rice, Master. Eugene', 'Williams, Mr. Charles Eugene', 'Vander Planke, Mrs. Julius (Emelia Maria Vandemoortele)',
            'Masselmani, Mrs. Fatima'
        ],
        'Sex': ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'male', 'female', 'female', 'female', 'female', 'male', 'male', 'male', 'female', 'male', 'male', 'female', 'female'],
        'Age': [22.0, 38.0, 26.0, 35.0, 35.0, np.nan, 54.0, 2.0, 27.0, 14.0, 4.0, 58.0, 20.0, 39.0, 14.0, 55.0, 2.0, np.nan, 31.0, np.nan],
        'SibSp': [1, 1, 0, 1, 0, 0, 0, 3, 0, 1, 1, 0, 0, 1, 0, 0, 4, 0, 1, 0],
        'Parch': [0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 1, 0, 0, 5, 0, 0, 1, 0, 0, 0],
        'Ticket': ['A/5 21171', 'PC 17599', 'STON/O2. 3101282', '113803', '373450', '330877', '17463', '349909', '347742', '237736', 'PP 9549', '113783', 'A/5. 2151', '347082', '350406', '248706', '382652', '244373', '345763', '2649'],
        'Fare': [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583, 51.8625, 21.075, 11.1333, 30.0708, 16.7, 26.55, 8.05, 31.275, 7.8542, 16.0, 29.125, 13.0, 18.0, 7.225],
        'Cabin': [np.nan, 'C85', np.nan, 'C123', np.nan, np.nan, 'E46', np.nan, np.nan, np.nan, 'G6', 'C103', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        'Embarked': ['S', 'C', 'S', 'S', 'S', 'Q', 'S', 'S', 'S', 'C', 'S', 'S', 'S', 'S', 'S', 'S', 'Q', 'S', 'S', 'C']
    }
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)

def create_mock_diabetes(filepath):
    """Create a miniature fallback mock Diabetes dataset."""
    print("Creating mock Diabetes dataset...")
    np.random.seed(42)
    n_samples = 200
    data = {
        'Pregnancies': np.random.randint(0, 10, n_samples),
        'Glucose': np.random.randint(70, 190, n_samples),
        'BloodPressure': np.random.randint(50, 110, n_samples),
        'SkinThickness': np.random.randint(10, 50, n_samples),
        'Insulin': np.random.randint(15, 300, n_samples),
        'BMI': np.round(np.random.uniform(18.0, 42.0, n_samples), 1),
        'DiabetesPedigreeFunction': np.round(np.random.uniform(0.1, 1.5, n_samples), 3),
        'Age': np.random.randint(21, 75, n_samples)
    }
    # Create simple probabilistic outcome based on Glucose and BMI
    prob = 1 / (1 + np.exp(-(0.04 * data['Glucose'] + 0.12 * data['BMI'] - 9)))
    data['Outcome'] = (np.random.rand(n_samples) < prob).astype(int)
    
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)

def prepare_all_datasets():
    """Load and prepare all datasets."""
    initialize_directories()
    download_titanic_dataset()
    download_diabetes_dataset()
    generate_sales_advertising_dataset()
    print("All datasets are successfully prepared!")

if __name__ == '__main__':
    prepare_all_datasets()

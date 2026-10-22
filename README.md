# Iris Flower Classification — CodeAlpha Internship

## Overview
This project trains a machine-learning classifier to predict an Iris flower species from four measurements:
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

The target column is `Species`. The `Id` column is an identifier and is excluded from model training.

## Dataset
The project uses the `Iris.csv` file supplied for the CodeAlpha internship task. It contains 150 rows and 6 columns. The dataset has three species labels: `Iris-setosa`, `Iris-versicolor`, and `Iris-virginica`.

## Methods
1. Load and inspect the dataset with pandas.
2. Check missing values and species distribution.
3. Visualize species counts and feature relationships.
4. Split data into training (80%) and test (20%) sets using stratification.
5. Train a pipeline with `StandardScaler` and `LogisticRegression`.
6. Evaluate the model using test accuracy, precision, recall, F1-score, and a confusion matrix.
7. Demonstrate a prediction for a sample flower.

## Setup and Run

### 1. Open the project folder in VS Code
Make sure `Iris.csv`, `iris_classification.py`, and `requirements.txt` are in the same folder.

### 2. Create and activate a virtual environment (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt instead:
```bat
.venv\Scripts\activate.bat
```

### 3. Install dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the project
```bash
python iris_classification.py
```

The script prints the dataset overview, test accuracy, classification report, and a sample prediction. It also creates an `outputs` folder containing:
- `species_distribution.png`
- `feature_relationships.png`
- `confusion_matrix.png`

## Notes
- The test score depends on the split and model. Do not claim a score until you run the script and see the actual output.
- This is a small educational dataset, so test accuracy is not a guarantee of performance on every real-world flower sample.

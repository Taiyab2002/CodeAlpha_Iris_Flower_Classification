# CodeAlpha Internship Task 1: Iris Flower Classification

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Keep the script and dataset in the same folder.
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Iris.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# 1. Load and inspect the dataset
df = pd.read_csv(DATA_PATH)

print("\n=== DATASET OVERVIEW ===")
print(f"Rows, columns: {df.shape}")
print("\nFirst five rows:")
print(df.head())
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nSpecies counts:")
print(df["Species"].value_counts())

# The Id column is only a row identifier, so it is not used as a model feature.
feature_columns = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
target_column = "Species"

X = df[feature_columns]
y = df[target_column]

# 2. Visualize the data and save charts for the project report
sns.set_theme(style="whitegrid")

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x=target_column, hue=target_column, legend=False)
plt.title("Number of Samples per Iris Species")
plt.xlabel("Species")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "species_distribution.png", dpi=150)
plt.close()

pair_plot = sns.pairplot(df, vars=feature_columns, hue=target_column, corner=True)
pair_plot.fig.suptitle("Iris Feature Relationships by Species", y=1.02)
pair_plot.savefig(OUTPUT_DIR / "feature_relationships.png", dpi=150)
plt.close("all")

# 3. Split the data into training and test sets.
# Stratification keeps the species proportions similar in both sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# 4. Train a classification pipeline.
# Scaling is fitted only on the training data to avoid data leakage.
model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ]
)
model.fit(X_train, y_train)

# 5. Evaluate the model on unseen test data.
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n=== MODEL EVALUATION ===")
print(f"Test accuracy: {accuracy:.2%}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)
plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
plt.close()

# 6. Demonstrate a prediction using four flower measurements (in cm).
sample_flower = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=feature_columns,
)
predicted_species = model.predict(sample_flower)[0]
print("\n=== SAMPLE PREDICTION ===")
print("Measurements (cm):", sample_flower.iloc[0].to_dict())
print("Predicted species:", predicted_species)

print(f"\nCharts saved in: {OUTPUT_DIR}")

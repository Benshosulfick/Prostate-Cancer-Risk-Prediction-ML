# ================= PROSTATE CANCER ML PROJECT (TCGA CLINICAL DATA) =================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# -------------------------------------------------------------------
# 1. LOAD DATASET
# -------------------------------------------------------------------

print("\nLoading TCGA Prostate Cancer Clinical Dataset...\n")

data = pd.read_csv("prad_broad_clinical_data.tsv", sep="\t")

# -------------------------------------------------------------------
# 2. SELECT IMPORTANT MEDICAL FEATURES
# -------------------------------------------------------------------

data = data[[
    "Diagnosis Age",
    "Serum PSA",
    "Tumor Stage",
    "Radical Prostatectomy Gleason Score for Prostate Cancer"
]]

data.columns = ["age", "psa", "stage", "gleason"]

# remove missing patients
data = data.dropna()

print("Total Patients after cleaning:", len(data))

# -------------------------------------------------------------------
# 3. CREATE MEDICAL TARGET (VERY IMPORTANT)
# -------------------------------------------------------------------
# Gleason score ≥ 7 means aggressive cancer

# ---------------- FIX GLEASON SCORE ----------------

# Extract number from values like "3+4", "4+3", "7", "8"
data["gleason"] = data["gleason"].astype(str)

def convert_gleason(x):
    try:
        if "+" in x:
            a, b = x.split("+")
            return int(a) + int(b)
        else:
            return int(float(x))
    except:
        return np.nan

data["gleason"] = data["gleason"].apply(convert_gleason)

# remove invalid rows
data = data.dropna()

print("Patients after Gleason cleaning:", len(data))

# Create target (clinical definition)
# Gleason >= 7 = aggressive prostate cancer
data["cancer_severity"] = (data["gleason"] >= 7).astype(int)

print("High Risk (Aggressive):", data["cancer_severity"].sum())
print("Low Risk:", len(data) - data["cancer_severity"].sum())

print("High Risk (Aggressive Cancer):", data["cancer_severity"].sum())
print("Low Risk :", len(data) - data["cancer_severity"].sum())

# -------------------------------------------------------------------
# 4. CONVERT TUMOR STAGE TO NUMERIC
# -------------------------------------------------------------------

# Convert stage to numeric automatically
data["stage"] = data["stage"].astype("category").cat.codes

# -------------------------------------------------------------------
# 5. FEATURES AND LABEL
# -------------------------------------------------------------------

# Remove gleason because it defines the label
X = data.drop(columns=["cancer_severity", "gleason"])
y = data["cancer_severity"]

# -------------------------------------------------------------------
# 6. TRAIN TEST SPLIT
# -------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------------------------------------------
# 7. MODELS
# -------------------------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier()
}

results = []

# -------------------------------------------------------------------
# 8. TRAIN & EVALUATE
# -------------------------------------------------------------------

print("\n================ MODEL PERFORMANCE ================\n")

for name, model in models.items():

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("MODEL:", name)
    print("Accuracy :", round(acc*100,2), "%")
    print("Precision:", round(prec*100,2), "%")
    print("Recall   :", round(rec*100,2), "%")
    print("F1 Score :", round(f1*100,2), "%")
    print("Confusion Matrix:\n", cm)
    print("--------------------------------------------------")

    results.append([name, acc*100, prec*100, rec*100, f1*100])

# -------------------------------------------------------------------
# 9. SAVE FINAL TABLE
# -------------------------------------------------------------------

results_df = pd.DataFrame(results, columns=[
    "Model", "Accuracy", "Precision", "Recall", "F1 Score"
])

results_df.to_csv("prostate_model_comparison.csv", index=False)

print("\nFinal comparison table saved as: prostate_model_comparison.csv")
print("\nDONE SUCCESSFULLY ✔")

# ================= CONFUSION MATRIX GRAPH =================

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix Values:\n", cm)

# Labels (important for report)
labels = ["Low Risk (Non-Aggressive)", "High Risk (Aggressive)"]

# Plot heatmap
plt.figure(figsize=(6,5))
sns.heatmap(cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels)

plt.title("Confusion Matrix – Prostate Cancer Prediction")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()

# SAVE IMAGE (IMPORTANT)
plt.savefig("confusion2_matrix_latest_dataset.png", dpi=300)

plt.show()

# =========================
# SORTED ACCURACY GRAPH
# =========================

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

results = {}

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Naive Bayes": GaussianNB(),
    "SVM (RBF)": SVC(kernel='rbf'),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=150),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# Train + store accuracies
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print(f"{name} Accuracy: {acc:.4f}")

# -------- SORT HIGH → LOW --------
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

model_names = [x[0] for x in sorted_results]
accuracies = [x[1] for x in sorted_results]

# -------- PLOT --------
plt.figure(figsize=(10,5))
bars = plt.bar(model_names, accuracies)

plt.xticks(rotation=40, ha='right')
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison (Highest to Lowest)")

# numbers on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2f}", ha='center')

plt.tight_layout()
plt.savefig("accuracy2_graph_sorted.png", dpi=300)
plt.show()

# =========================
# PRECISION vs RECALL GRAPH
# =========================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score

precision_list = []
recall_list = []
model_labels = []

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=150),
    "SVM (RBF)": SVC(kernel='rbf'),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5)
}

# Train and calculate metrics
for name, model in models.items():

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)

    precision_list.append(precision)
    recall_list.append(recall)
    model_labels.append(name)

    print(f"{name} -> Precision: {precision:.4f}  Recall: {recall:.4f}")

# ----- plotting -----
x = np.arange(len(model_labels))
width = 0.35

plt.figure(figsize=(11,5))

# SAME COLORS AS PAPER
bars1 = plt.bar(x - width/2, precision_list, width, label='Precision (Macro Avg)', color='#1f77b4')  # blue
bars2 = plt.bar(x + width/2, recall_list, width, label='Recall (Macro Avg)', color='#ff7f0e')       # orange

plt.xticks(x, model_labels, rotation=35, ha='right')
plt.ylabel("Score")
plt.title("Precision vs Recall Comparison (Overall / Macro Average)")
plt.legend()

# show values on bars
for bars in [bars1, bars2]:
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f"{yval:.2f}", ha='center', fontsize=9)

plt.tight_layout()
plt.savefig("precision2_recall_graph.png", dpi=300)
plt.show()
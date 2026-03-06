# ================= PROSTATE CANCER ML PROJECT (MSKCC CLINICAL DATASET) =================

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

print("\nLoading MSKCC Clinical Prostate Dataset...\n")

# ---------- LOAD DATASET ----------
data = pd.read_csv("prad_mskcc_clinical_data.tsv", sep="\t")

# ================= FIND GLEASON COLUMN =================

gleason_col = None

for col in data.columns:
    if "gleason" in col.lower():
        gleason_col = col
        break

if gleason_col is None:
    raise ValueError("No Gleason score column found in dataset!")

print("Using Gleason column:", gleason_col)

# convert to numeric
data[gleason_col] = pd.to_numeric(data[gleason_col], errors="coerce")

# remove missing gleason rows
data = data.dropna(subset=[gleason_col])

# create target
data["cancer_severity"] = (data[gleason_col] >= 7).astype(int)

print("High Risk:", data["cancer_severity"].sum())
print("Low Risk:", len(data) - data["cancer_severity"].sum())

# ================= REMOVE LEAKAGE (BUT KEEP GLEASON) =================

bad_words = [
    "stage", "ajcc", "pathologic", "clinical",
    "grade", "risk", "tumor", "treatment",
    "metastasis", "lymph", "node", "status",
    "recurrence", "survival"
]

cols_to_drop = []

for col in data.columns:
    name = col.lower()
    if any(word in name for word in bad_words):
        if col != gleason_col:   # <-- VERY IMPORTANT
            cols_to_drop.append(col)

print("Dropping leakage columns:")
print(cols_to_drop)

data.drop(columns=cols_to_drop, inplace=True, errors="ignore")

print("Total Patients:", len(data))

# ===== REMOVE DATA LEAKAGE COLUMNS =====

leakage_columns = [
    "Overall Survival (Months)",
    "Overall Survival Status",
    "Disease Free (Months)",
    "Disease Free Status",
    "Biochemical Recurrence",
    "Time to Biochemical Recurrence",
    "PSA Doubling Time",
    "Metastasis",
    "Recurrence",
    "Followup",
    "Cause of Death"
]

for col in leakage_columns:
    if col in data.columns:
        data.drop(columns=col, inplace=True)

print("Leakage columns removed")

# ---------- FIND GLEASON COLUMN AUTOMATICALLY ----------
gleason_col = None
for col in data.columns:
    if "gleason" in col.lower():
        gleason_col = col
        break

if gleason_col is None:
    raise ValueError("No Gleason score column found in dataset!")

print("Using Gleason column:", gleason_col)

# ---------- CLEAN DATA ----------
# convert to numeric
data[gleason_col] = pd.to_numeric(data[gleason_col], errors="coerce")

# remove missing gleason
data = data.dropna(subset=[gleason_col])

# ---------- CREATE TARGET ----------
# Aggressive (1) if Gleason >= 7
data["cancer_severity"] = (data[gleason_col] >= 7).astype(int)
X = data.copy()
y = X.pop("cancer_severity")

# also remove the original gleason from features
X.drop(columns=[gleason_col], inplace=True, errors="ignore")

print("High Risk:", data["cancer_severity"].sum())
print("Low Risk :", len(data) - data["cancer_severity"].sum())

# ========= KEEP ONLY NUMERIC FEATURES =========

# remove label first
X = data.drop(columns=["cancer_severity"], errors="ignore")

# remove original Gleason column
X = X.drop(columns=[gleason_col], errors="ignore")

# keep only numeric columns
X = X.select_dtypes(include=[np.number])

# ===== REMOVE DATA LEAKAGE FEATURES =====

leak_keywords = [
    "gleason",
    "grade",
    "stage",
    "tumor",
    "pathologic",
    "margin",
    "t_stage",
    "n_stage",
    "m_stage"
]

cols_to_remove = [col for col in X.columns
                  if any(word in col.lower() for word in leak_keywords)]

print("Removed leakage columns:", cols_to_remove)

X = X.drop(columns=cols_to_remove, errors="ignore")

# target
y = data["cancer_severity"]

# ---------------- HANDLE MISSING VALUES ----------------
print("Missing values before cleaning:", X.isnull().sum().sum())

# replace missing numeric values with median
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)

print("Missing values after cleaning: 0")

# print("Numeric features kept:", len(X.columns))
print("Number of features used:", X.shape[1])

# ---------- TRAIN TEST SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ---- BALANCE THE DATASET ----
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------- MODELS ----------
models = {
    "Logistic Regression": LogisticRegression(max_iter=500, class_weight="balanced"),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced"),
    "SVM": SVC(kernel="rbf", class_weight="balanced"),
    "KNN": KNeighborsClassifier(n_neighbors=7)
}

results = []

print("\n================ MODEL PERFORMANCE ================\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred) * 100
    rec = recall_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred) * 100
    cm = confusion_matrix(y_test, y_pred)

    print("MODEL:", name)
    print("Accuracy :", round(acc,2), "%")
    print("Precision:", round(prec,2), "%")
    print("Recall   :", round(rec,2), "%")
    print("F1 Score :", round(f1,2), "%")
    print("Confusion Matrix:\n", cm)
    print("------------------------------------")

    results.append([name, acc, prec, rec, f1])

# ---------- SAVE TABLE ----------
results_df = pd.DataFrame(results, columns=["Model","Accuracy","Precision","Recall","F1 Score"])
results_df.to_csv("MSKCC_model_comparison.csv", index=False)

print("\nFINAL TABLE SAVED -> MSKCC_model_comparison.csv")
print("DONE SUCCESSFULLY 🎯")

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
plt.savefig("confusion3_matrix_latest_dataset.png", dpi=300)

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
plt.savefig("accuracy3_graph_sorted.png", dpi=300)
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
plt.savefig("precision3_recall_graph.png", dpi=300)
plt.show()
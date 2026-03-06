import pandas as pd
import numpy as np
import requests
from io import StringIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

print("Loading Clinical Prostate Dataset...\n")

data = pd.read_csv("prostate 3.csv")


data = data.drop(columns=["train"])

print("Total Patients:", len(data))
print("Columns:", data.columns)

psa = data["lpsa"]

X = data.drop(columns=["lpsa"])

median_psa = psa.median()
y = (psa > median_psa).astype(int)

print("Median PSA:", round(median_psa,2))
print("Low Risk:", (y==0).sum())
print("High Risk:", (y==1).sum())

X = data.drop(columns=["lpsa"])
psa = data["lpsa"]


median_psa = psa.median()
y = (psa > median_psa).astype(int)

print("Patients:", len(y))
print("Features:", X.shape[1])
print("Low risk:", (y==0).sum())
print("High risk:", (y==1).sum())

#
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier()
}

print("\n========== MODEL PERFORMANCE ==========\n")
results = []

for name, model in models.items():

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\nMODEL:", name)
    print("Accuracy :", round(acc*100,2), "%")
    print("Precision:", round(prec*100,2), "%")
    print("Recall   :", round(rec*100,2), "%")
    print("F1 Score :", round(f1*100,2), "%")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # store results
    results.append([name, acc, prec, rec, f1])

# Create comparison table
results_df = pd.DataFrame(results, columns=[
    "Model", "Accuracy", "Precision", "Recall", "F1 Score"
])

# convert to percentage
results_df[["Accuracy","Precision","Recall","F1 Score"]] *= 100

print("\n================ FINAL MODEL COMPARISON TABLE ================\n")
print(results_df)

# Save table
results_df.to_csv("prostate_model_comparison.csv", index=False)
print("\nSaved as prostate_model_comparison.csv")

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
plt.savefig("confusion1_matrix_latest_dataset.png", dpi=300)

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
plt.savefig("accuracy1_graph_sorted.png", dpi=300)
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
plt.savefig("precision1_recall_graph.png", dpi=300)
plt.show()
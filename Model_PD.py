import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import joblib


# =========================
# DATASET
# =========================

df = pd.read_csv("heart.csv")

print(df.head())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())


# =========================
# CORRELATION
# =========================

target_corr = df.corr(numeric_only=True)[['target']].sort_values(
    by='target',
    ascending=False
)

print("\nTarget Correlation:")
print(target_corr)


# =========================
# DUPLICATES
# =========================

print("\nDuplicates:", df.duplicated().sum())

df = df.drop_duplicates()


# =========================
# X AND Y
# =========================

X = df.drop(['target'], axis=1)
y = df['target']


# =========================
# TRAIN-TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# FEATURE SCALING
# =========================

std = StandardScaler()

X_train = X_train.copy()
X_test = X_test.copy()

X_train_Scaled = std.fit_transform(X_train)
X_test_Scaled = std.transform(X_test)


# =========================
# MODELS
# =========================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC()
}


# =========================
# TRAINING & EVALUATION
# =========================

results = []

for name, model in models.items():

    model.fit(X_train_Scaled, y_train)

    y_pred = model.predict(X_test_Scaled)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append({
        'Model': name,
        'Accuracy': round(acc, 4),
        'F1 Score': round(f1, 4)
    })

    print(f"\n========== {name} ==========")
    print(classification_report(y_test, y_pred))


# =========================
# RESULTS COMPARISON
# =========================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by='Accuracy',
    ascending=False
)

print("\nModel Comparison:")
print(results_df)


# =========================
# VISUALIZATION
# =========================

plt.figure(figsize=(10, 5))

sns.barplot(
    data=results_df,
    x='Accuracy',
    y='Model'
)

plt.title("Heart Disease Model Comparison")
plt.xlabel("Accuracy")
plt.ylabel("ML Model")
plt.tight_layout()
plt.show()


# =========================
# SAVE BEST MODEL
# =========================

best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]

print("\nBest Model:", best_model_name)

joblib.dump(best_model, 'Best_Model.pkl')
joblib.dump(std, 'Scaled.pkl')
joblib.dump(X.columns.to_list(), 'columns.pkl')

print("\nModel, scaler and columns saved successfully!")
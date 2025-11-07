# Bank churn classifier (neural network using scikit-learn MLP)
# Satisfies: read dataset, split features/target, normalize, build model,
# implement improvements (handle class imbalance by upsampling minority; enable early stopping),
# print accuracy and confusion matrix.

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import resample

RND = 42

# 1. Read dataset (local preferred; fallback to public mirror)
path = "Churn_Modelling.csv"
df = pd.read_csv(path)


# 2. Feature/target split and basic preprocessing
if 'Exited' not in df.columns:
    raise RuntimeError("Dataset must contain 'Exited' column (target).")

df = df.copy()
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df = pd.get_dummies(df, columns=['Geography'], drop_first=True)
df = df.drop(columns=[c for c in ['CustomerId', 'Surname', 'RowNumber'] if c in df.columns])

X = df.drop(columns=['Exited'])
y = df['Exited'].astype(int)

# stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RND, stratify=y
)

# 3. Normalize train and test
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Identify improvement: class imbalance handling -> upsample minority class in training set
# Combine X_train and y_train to perform resampling
train_df = pd.DataFrame(X_train, columns=X.columns)
train_df['Exited'] = y_train.values

majority = train_df[train_df['Exited'] == 0]
minority = train_df[train_df['Exited'] == 1]

if len(minority) == 0:
    # if no minority present (unexpected), skip resampling
    X_train_res = X_train
    y_train_res = y_train
else:
    minority_upsampled = resample(minority,
                                  replace=True,
                                  n_samples=len(majority),
                                  random_state=RND)
    train_balanced = pd.concat([majority, minority_upsampled])
    train_balanced = train_balanced.sample(frac=1, random_state=RND).reset_index(drop=True)
    y_train_res = train_balanced['Exited'].astype(int)
    X_train_res = train_balanced.drop(columns=['Exited']).values

# 4. Initialize and build the model (MLP with early stopping)
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=500,
    early_stopping=True,
    n_iter_no_change=20,
    tol=1e-4,
    random_state=RND
)

# Train on the (possibly) resampled training set
mlp.fit(X_train_res, y_train_res)

# 5. Evaluate: print accuracy score and confusion matrix
y_pred = mlp.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {acc:.4f}")
print("Confusion matrix:")
print(cm)

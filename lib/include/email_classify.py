# Single Jupyter cell — robust Email Spam classification (KNN and SVM)
# Put this file (emails.csv) in the same folder as the notebook or give a path.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings("ignore")

# 1) Load dataset
df = pd.read_csv("emails.csv")
print("Dataset loaded. Shape:", df.shape)
print("Columns:", list(df.columns)[:12], "..." if len(df.columns)>12 else "")

# 2) Heuristic: find label column and feature source
# Try common label names first
label_candidates = ['label','spam','class','target','is_spam']
label_col = None
for c in label_candidates:
    if c in df.columns:
        label_col = c
        break

# If not found, try to infer: numeric column with only 0/1 or strings 'spam','ham'
if label_col is None:
    for c in df.columns:
        vals = df[c].dropna().unique()
        if set(vals).issubset({0,1}) or set(map(str.lower, map(str, vals))).intersection({'spam','ham','spam','not spam','ham','ham\n'}):
            label_col = c
            break

if label_col is None:
    raise ValueError("Could not find label column automatically. Please set `label_col` manually (e.g. 'label' or 'spam').")

print("Detected label column:", label_col)
y = df[label_col].copy()
# Normalize label values to 0/1 if needed
y = y.map(lambda v: 1 if str(v).strip().lower() in ('1','spam','true','t','yes') else 0)

# 3) Decide features: text -> TF-IDF, else numeric columns -> use as-is
text_col = None
# common text column names
for c in ['text','email','content','message','body','Email']:
    if c in df.columns and df[c].dtype == object:
        text_col = c
        break

# If many object columns, prefer the longest text column
if text_col is None:
    obj_cols = [c for c in df.columns if df[c].dtype == object and c!=label_col]
    if obj_cols:
        # choose the one with highest avg length
        lengths = {c: df[c].astype(str).map(len).mean() for c in obj_cols}
        text_col = max(lengths, key=lengths.get)

if text_col:
    print("Detected text column for vectorization:", text_col)
    X_text = df[text_col].fillna("").astype(str)
    # TF-IDF vectorizer — keep a reasonable max features to avoid memory spikes
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))
    X = tfidf.fit_transform(X_text)
    feature_type = "tfidf"
else:
    # Use numeric columns (exclude label and non-numeric)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # remove label if present in numeric_cols
    numeric_cols = [c for c in numeric_cols if c != label_col]
    if len(numeric_cols) == 0:
        raise ValueError("No text column found and no numeric features detected. Please prepare features.")
    print(f"Using numeric feature columns (count={len(numeric_cols)}).")
    X = df[numeric_cols].fillna(0).values
    feature_type = "numeric"

print("Feature type:", feature_type)

# 4) Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
print("Train/test split:", X_train.shape, X_test.shape, "Labels distribution (train):", np.bincount(y_train.astype(int)))

# 5) Define and train models
results = {}

# Helper to fit and evaluate (handles sparse/dense inputs)
def fit_and_eval(name, model, Xtr, Xte, ytr, yte, scale_numeric=True):
    # If features are numeric and scaling helps, use scaler pipeline
    if feature_type == "numeric" and scale_numeric:
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', model)])
        pipe.fit(Xtr, ytr)
        ypred = pipe.predict(Xte)
        try:
            probas = pipe.predict_proba(Xte)[:,1]
        except:
            probas = None
    else:
        # TF-IDF or non-scaling case
        model.fit(Xtr, ytr)
        ypred = model.predict(Xte)
        try:
            probas = model.predict_proba(Xte)[:,1]
        except:
            probas = None

    acc = accuracy_score(yte, ypred)
    cls_report = classification_report(yte, ypred, digits=4)
    cm = confusion_matrix(yte, ypred)
    roc = None
    if probas is not None and len(np.unique(yte))==2:
        try:
            roc = roc_auc_score(yte, probas)
        except:
            roc = None
    results[name] = dict(accuracy=acc, report=cls_report, confusion=cm, roc_auc=roc)
    print(f"\n=== {name} ===")
    print("Accuracy:", acc)
    if roc is not None:
        print("ROC AUC:", roc)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", cls_report)

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=7)
fit_and_eval("K-Nearest Neighbors (k=7)", knn, X_train, X_test, y_train, y_test)

# Support Vector Machine (RBF)
# For efficiency on TF-IDF sparse data, use probability=False by default (probabilities cost more)
svm = SVC(C=1.0, kernel='rbf', gamma='scale', probability=True)
fit_and_eval("Support Vector Machine (RBF)", svm, X_train, X_test, y_train, y_test, scale_numeric=False)

# 6) Summary comparison
print("\n\n=== Summary Comparison ===")
for k,v in results.items():
    print(f"{k}: Accuracy={v['accuracy']:.4f}" + (f", ROC AUC={v['roc_auc']:.4f}" if v['roc_auc'] else ""))

# End of single-cell script.

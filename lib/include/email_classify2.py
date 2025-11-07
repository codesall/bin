import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
df = pd.read_csv("emails.csv")

df.head()

df.isnull().sum()

print(df.head())

X = df.iloc[:,1:3001]  # word frequency features
X

Y = df.iloc[:,-1].values # 1 = spam, 0 = not spam
Y

# Split data 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

from sklearn.metrics import classification_report, confusion_matrix

# -------- Support Vector Machine --------
svc = SVC(C=1.0, kernel='rbf', gamma='auto')
svc.fit(X_train, y_train)
svc_pred = svc.predict(X_test)

print("SVM Accuracy:", accuracy_score(y_test, svc_pred))
print("SVM Classification Report:\n", classification_report(y_test, svc_pred))
print("SVM Confusion Matrix:\n", confusion_matrix(y_test, svc_pred))

# -------- K-Nearest Neighbors --------
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

print("KNN Accuracy:", knn.score(X_test, y_test))
print("KNN Classification Report:\n", classification_report(y_test, knn_pred))
print("KNN Confusion Matrix:\n", confusion_matrix(y_test, knn_pred))

def predict_email(new_email_vector):
    """
    Predicts whether a given email is spam or not.
    Input: new_email_vector - array/list of 3000 features (same structure as dataset)
    Output: Prints prediction result for both models
    """
    test_df = pd.DataFrame([new_email_vector], columns=X.columns)

    pred_svm = svc.predict(test_df)[0]
    pred_knn = knn.predict(test_df)[0]

    print("\n===== Custom Email Prediction =====")
    print(f"SVM Prediction: {'Spam' if pred_svm == 1 else 'Not Spam'}")
    print(f"KNN Prediction: {'Spam' if pred_knn == 1 else 'Not Spam'}")


# ✅ Example: test a sample email feature vector (outside the function)
sample_email = X_test.iloc[0].values  # pick the first test sample
predict_email(sample_email)


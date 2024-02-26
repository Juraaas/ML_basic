import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Load your dataset from the 'australian.csv' file
data = pd.read_csv('australian.csv')

# Preprocess the data
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# Decision Tree Classifier
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_y_pred = dt_model.predict(X_test)

# K-Nearest Neighbors (KNN) Classifier
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_y_pred = knn_model.predict(X_test)

# Logistic Regression
lr_model = LogisticRegression(max_iter=10000)
lr_model.fit(X_train, y_train)
lr_y_pred = lr_model.predict(X_test)

# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)

# Naive Bayes (Gaussian) Classifier
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_y_pred = nb_model.predict(X_test)

# Calculate the confusion matrix and evaluation metrics for Decision Tree Classifier
dt_confusion = confusion_matrix(y_test, dt_y_pred)
dt_accuracy = accuracy_score(y_test, dt_y_pred)
dt_precision = precision_score(y_test, dt_y_pred)
dt_recall = recall_score(y_test, dt_y_pred)
dt_f1 = f1_score(y_test, dt_y_pred)

# Calculate the confusion matrix and evaluation metrics for K-Nearest Neighbors (KNN) Classifier
knn_confusion = confusion_matrix(y_test, knn_y_pred)
knn_accuracy = accuracy_score(y_test, knn_y_pred)
knn_precision = precision_score(y_test, knn_y_pred)
knn_recall = recall_score(y_test, knn_y_pred)
knn_f1 = f1_score(y_test, knn_y_pred)

# Calculate the confusion matrix and evaluation metrics for Logistic Regression
lr_confusion = confusion_matrix(y_test, lr_y_pred)
lr_accuracy = accuracy_score(y_test, lr_y_pred)
lr_precision = precision_score(y_test, lr_y_pred)
lr_recall = recall_score(y_test, lr_y_pred)
lr_f1 = f1_score(y_test, lr_y_pred)

# Calculate the confusion matrix and evaluation metrics for Random Forest Classifier
rf_confusion = confusion_matrix(y_test, rf_y_pred)
rf_accuracy = accuracy_score(y_test, rf_y_pred)
rf_precision = precision_score(y_test, rf_y_pred)
rf_recall = recall_score(y_test, rf_y_pred)
rf_f1 = f1_score(y_test, rf_y_pred)

# Calculate the confusion matrix and evaluation metrics for Naive Bayes Classifier
nb_confusion = confusion_matrix(y_test, nb_y_pred)
nb_accuracy = accuracy_score(y_test, nb_y_pred)
nb_precision = precision_score(y_test, nb_y_pred)
nb_recall = recall_score(y_test, nb_y_pred)
nb_f1 = f1_score(y_test, nb_y_pred)

# Print the results for Decision Tree Classifier
print('\nDecision Tree Classifier:')
print('Confusion Matrix:')
print(dt_confusion)
print('Accuracy:', dt_accuracy)
print('Precision:', dt_precision)
print('Recall:', dt_recall)
print('F1 Score:', dt_f1)

# Print the results for K-Nearest Neighbors (KNN) Classifier
print('\nK-Nearest Neighbors (KNN) Classifier:')
print('Confusion Matrix:')
print(knn_confusion)
print('Accuracy:', knn_accuracy)
print('Precision:', knn_precision)
print('Recall:', knn_recall)
print('F1 Score:', knn_f1)

# Print the results for Logistic Regression
print('\nLogistic Regression:')
print('Confusion Matrix:')
print(lr_confusion)
print('Accuracy:', lr_accuracy)
print('Precision:', lr_precision)
print('Recall:', lr_recall)
print('F1 Score:', lr_f1)

# Print the results for Random Forest Classifier
print('\nRandom Forest Classifier:')
print('Confusion Matrix:')
print(rf_confusion)
print('Accuracy:', rf_accuracy)
print('Precision:', rf_precision)
print('Recall:', rf_recall)
print('F1 Score:', rf_f1)

# Print the results for Naive Bayes Classifier
print('\nNaive Bayes (Gaussian):')
print('Confusion Matrix:')
print(nb_confusion)
print('Accuracy:', nb_accuracy)
print('Precision:', nb_precision)
print('Recall:', nb_recall)
print('F1 Score:', nb_f1)
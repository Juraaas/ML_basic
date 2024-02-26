import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils import compute_sample_weight

# Load the data
input_file = 'D:/cs4/australian.csv'
output_file = 'australian.csv'

data_lines = []

with open(input_file, 'r') as file:
    lines = file.readlines()
    start_data = False

    for line in lines:
        if start_data:
            fields = line.strip().split()
            data_lines.append(fields)
        elif line.strip() == "OBJECTS 690":
            start_data = True

header = [
    "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11",
    "A12", "A13", "A14", "CLASS"
]

df = pd.DataFrame(data_lines, columns=header)

df["CLASS"] = df["CLASS"].replace({'+': 1, '-': 0})

df.to_csv(output_file, index=False)

data = pd.read_csv(output_file)
X = data.iloc[:, -1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

class WeightedKNN(KNeighborsClassifier):
    def fit(self, X, y, sample_weight):
        self.sample_weight = sample_weight
        return super().fit(X, y)

    def predict(self, X):
        return super().predict(X)

sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
weighted_knn = WeightedKNN(n_neighbors=5)
weighted_knn.fit(X_train, y_train, sample_weight=sample_weights)

adaboost_model = AdaBoostClassifier(estimator=weighted_knn, n_estimators=50)
adaboost_model.fit(X_train, y_train)

y_pred = adaboost_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

print('\nAdaBoost with KNN:')
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)
print('Confusion Matrix:')
print(confusion)
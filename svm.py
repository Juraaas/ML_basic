import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

input_file = 'D:/cs/australian.tab'
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

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# Create an SVM model with an RBF kernel
svm_model = SVC(kernel='rbf')
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

confusion = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print('Confusion Matrix:')
print(confusion)

print('Precision:', precision)
print('Recall:', recall)
print('Accuracy:', accuracy)
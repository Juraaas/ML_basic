import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.exceptions import UndefinedMetricWarning
import warnings

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

breast_cancer = load_breast_cancer()
X, y = breast_cancer.data, breast_cancer.target

num_iterations = 1
subset_size_objects = 200
subset_size_attributes = 15

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
classifier_names = ['SVM', 'Naive Bayes', 'Decision Tree']
committee_names = ['Committee (k-NN + SVM)', 'Committee (k-NN + SVM + Naive Bayes)', 'Committee (All)']
algorithm_names = ['k-NN'] + classifier_names + committee_names
performance_data = {metric: {algo: [] for algo in algorithm_names} for metric in metrics_list}

predictions_df = pd.DataFrame(data={'True Label': y_test})

selected_objects = np.random.choice(X_train.shape[0], size=subset_size_objects, replace=False)
selected_attributes = np.random.choice(X_train.shape[1], size=min(subset_size_attributes, X_train.shape[1]), replace=True)
X_subset_train = X_train[selected_objects][:, selected_attributes]
y_subset_train = y_train[selected_objects]

knn = KNeighborsClassifier()
svm = SVC(probability=True)
nb = GaussianNB()
dt = DecisionTreeClassifier()

committee_knn_svm = VotingClassifier(estimators=[('knn', knn), ('svm', svm)], voting='soft')
committee_knn_svm_nb = VotingClassifier(estimators=[('knn', knn), ('svm', svm), ('nb', nb)], voting='soft')
committee_all = VotingClassifier(estimators=[('knn', knn), ('svm', svm), ('nb', nb), ('dt', dt)], voting='soft')

knn.fit(X_subset_train, y_subset_train)
svm.fit(X_subset_train, y_subset_train)
nb.fit(X_subset_train, y_subset_train)
dt.fit(X_subset_train, y_subset_train)
committee_knn_svm.fit(X_subset_train, y_subset_train)
committee_knn_svm_nb.fit(X_subset_train, y_subset_train)
committee_all.fit(X_subset_train, y_subset_train)

knn_pred = knn.predict(X_test[:, selected_attributes])
svm_pred = svm.predict(X_test[:, selected_attributes])
nb_pred = nb.predict(X_test[:, selected_attributes])
dt_pred = dt.predict(X_test[:, selected_attributes])
committee_knn_svm_pred = committee_knn_svm.predict(X_test[:, selected_attributes])
committee_knn_svm_nb_pred = committee_knn_svm_nb.predict(X_test[:, selected_attributes])
committee_all_pred = committee_all.predict(X_test[:, selected_attributes])

predictions_df['k-NN'] = knn_pred
predictions_df['SVM'] = svm_pred
predictions_df['Naive Bayes'] = nb_pred
predictions_df['Decision Tree'] = dt_pred

for committee_pred in [committee_knn_svm_pred, committee_knn_svm_nb_pred, committee_all_pred]:
    unique, counts = np.unique(committee_pred, return_counts=True)
    if len(unique) == 2 and counts[0] == counts[1]:
        print(f"Tie in Committee predictions. No decision made.")
        committee_pred[:] = np.nan

predictions_df['Committee (k-NN + SVM)'] = committee_knn_svm_pred
predictions_df['Committee (k-NN + SVM + Naive Bayes)'] = committee_knn_svm_nb_pred
predictions_df['Committee (All)'] = committee_all_pred

for algo in algorithm_names:
    pred = predictions_df[algo]
    if 'Committee' in algo:
        # Exclude instances with ties from committee predictions
        pred = pred[~pred.isna()]
    performance_data['Accuracy'][algo].append(accuracy_score(y_test, pred))
    performance_data['Precision'][algo].append(precision_score(y_test, pred))
    performance_data['Recall'][algo].append(recall_score(y_test, pred))
    performance_data['F1 Score'][algo].append(f1_score(y_test, pred))

pd.set_option('display.max_columns', None)
print("Predictions:")
print(predictions_df)

print("\nPerformance Metrics:")
performance_metrics_df = pd.DataFrame(performance_data)
print(performance_metrics_df)

plt.figure(figsize=(12, 6))
plt.title('wykres')

classifiers_line = ['SVM', 'Naive Bayes', 'Decision Tree']
plt.plot(range(len(classifiers_line) + 1),
         [performance_data['Accuracy']['k-NN'][0]] +
         [performance_data['Accuracy'][algo][0] for algo in classifiers_line],
         color='gray', linestyle='-', marker='o', label='Klasyfikatory')

# Plot connections between k-NN and committees
committees_line = ['Committee (k-NN + SVM)', 'Committee (k-NN + SVM + Naive Bayes)', 'Committee (All)']
plt.plot(range(len(classifiers_line) + 1),
         [performance_data['Accuracy']['k-NN'][0]] +
         [performance_data['Accuracy'][algo][0] for algo in committees_line],
         color='blue', linestyle='-', marker='o', label='Komitety')


plt.xlabel('Algorithm')
plt.ylabel('Accuracy')
plt.xticks(range(len(classifiers_line) + len(committees_line) + 1),
           ['k-NN'] + classifiers_line + committees_line, rotation=45)
plt.legend()
plt.show()
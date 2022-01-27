import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import seaborn as sn
import time
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

# Importing the dataset into a dataframe
data = pd.read_csv('Binary Dataset.csv')

X = data.iloc[:, 0: 10].values
Y = data.iloc[:, -1].values

# Splitting the dataset into testing and training subsets
training_X, testing_X, training_Y, testing_Y = train_test_split(X, Y, test_size = 0.3, random_state = 42)

# Scaling the data
sc = StandardScaler()
sc.fit(training_X, testing_X)
training_X_std = sc.transform(training_X)
testing_X_std = sc.transform(testing_X)


# Naive Bayes - Binary Classification

# Generate starting timestamp
start_time = time.time()

# Creating the classifier
NB_model = GaussianNB()
# Classifier is trained with training set
NB_model.fit(training_X_std, training_Y)

# Predicting testing set
predict_Y = NB_model.predict(testing_X_std)

# Generate finishing timestamp
finish_time = time.time()

# Creating the confusion matrix and performance metrics
cm = confusion_matrix(testing_Y, predict_Y)
acc = accuracy_score(testing_Y, predict_Y)
pre = precision_score(testing_Y, predict_Y)
rec = recall_score(testing_Y, predict_Y)
f_mea = f1_score(testing_Y, predict_Y)
process_time = finish_time - start_time

# Outputting confusion matrix
tick_labels = ["Benign", "Malicious"]
ax = sn.heatmap(cm / np.sum(cm), annot = True, fmt = '.2%', 
           xticklabels = tick_labels, yticklabels = tick_labels, cmap = 'Blues')
ax.set_title('Confusion Matrix for Naive Bayes classifier');
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
plt.show()

# Displaying perfomance metrics
print("Accuracy score: " + str(acc * 100) + "%")
print("Precision score: " + str(pre * 100) + "%")
print("Recall score: " + str(rec * 100) + "%")
print("f-measure score: " + str(f_mea * 100) + "%")
print("Processing time: " + str(process_time) + " seconds")


# Random Forest - Binary Classification

start_time = time.time()

RF_model = RandomForestClassifier(n_estimators = 18)

RF_model.fit(training_X_std, training_Y)

predict_Y = RF_model.predict(testing_X_std)

finish_time = time.time()

cm = confusion_matrix(testing_Y, predict_Y)
acc = accuracy_score(testing_Y, predict_Y)
pre = precision_score(testing_Y, predict_Y)
rec = recall_score(testing_Y, predict_Y)
f_mea = f1_score(testing_Y, predict_Y)
process_time = finish_time - start_time

tick_labels = ["Benign", "Malicious"]
ax = sn.heatmap(cm / np.sum(cm), annot = True, fmt = '.2%', 
           xticklabels = tick_labels, yticklabels = tick_labels, cmap = 'Blues')
ax.set_title('Confusion Matrix for Random Forest classifier');
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
plt.show()

print("Accuracy score: " + str(acc * 100) + "%")
print("Precision score: " + str(pre * 100) + "%")
print("Recall score: " + str(rec * 100) + "%")
print("f-measure score: " + str(f_mea * 100) + "%")
print("Processing time: " + str(process_time) + " seconds")


# k Nearest Neighbours - Binary Classification

start_time = time.time()

kNN_model = KNeighborsClassifier(n_neighbors = 4)

kNN_model.fit(training_X_std, training_Y)

predict_Y = kNN_model.predict(testing_X_std)

finish_time = time.time()

cm = confusion_matrix(testing_Y, predict_Y)
acc = accuracy_score(testing_Y, predict_Y)
pre = precision_score(testing_Y, predict_Y)
rec = recall_score(testing_Y, predict_Y)
f_mea = f1_score(testing_Y, predict_Y)
process_time = finish_time - start_time

tick_labels = ["Benign", "Malicious"]
ax = sn.heatmap(cm / np.sum(cm), annot = True, fmt = '.2%', 
           xticklabels = tick_labels, yticklabels = tick_labels, cmap = 'Blues')
ax.set_title('Confusion Matrix for KNN classifier');
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
plt.show()

print("Accuracy score: " + str(acc * 100) + "%")
print("Precision score: " + str(pre * 100) + "%")
print("Recall score: " + str(rec * 100) + "%")
print("f-measure score: " + str(f_mea * 100) + "%")
print("Processing time: " + str(process_time) + " seconds")


# Support Vector Machines - Binary Classification

start_time = time.time()

SVM_model = svm.SVC(kernel='poly')

SVM_model.fit(training_X_std, training_Y)

predict_Y = SVM_model.predict(testing_X_std)

finish_time = time.time()

cm = confusion_matrix(testing_Y, predict_Y)
acc = accuracy_score(testing_Y, predict_Y)
pre = precision_score(testing_Y, predict_Y)
rec = recall_score(testing_Y, predict_Y)
f_mea = f1_score(testing_Y, predict_Y)
process_time = finish_time - start_time

tick_labels = ["Benign", "Malicious"]
ax = sn.heatmap(cm / np.sum(cm), annot = True, fmt = '.2%', 
           xticklabels = tick_labels, yticklabels = tick_labels, cmap = 'Blues')
ax.set_title('Confusion Matrix for SVM classifier');
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
plt.show()

print("Accuracy score: " + str(acc * 100) + "%")
print("Precision score: " + str(pre * 100) + "%")
print("Recall score: " + str(rec * 100) + "%")
print("f-measure score: " + str(f_mea * 100) + "%")
print("Processing time: " + str(process_time) + " seconds")


# Changing dataset to only malicious traffic
data = pd.read_csv('Malicious Dataset.csv')

X = data.iloc[:, 0: 10].values
Y = data.iloc[:, -1].values

# Splitting the dataset into testing and training subsets
training_X, testing_X, training_Y, testing_Y = train_test_split(X, Y, test_size = 0.3, random_state = 42)

# Scaling the data
sc = StandardScaler()
sc.fit(training_X, testing_X)
training_X_std = sc.transform(training_X)
testing_X_std = sc.transform(testing_X)


# Naive Bayes - Multiclass Classification

start_time = time.time()

NB_model = MultinomialNB()

NB_model.fit(training_X, training_Y)

predict_Y = NB_model.predict(testing_X)

finish_time = time.time()

cm = confusion_matrix(testing_Y, predict_Y)
acc = accuracy_score(testing_Y, predict_Y)
pre = precision_score(testing_Y, predict_Y, average = 'weighted')
rec = recall_score(testing_Y, predict_Y, average = 'weighted')
f_mea = f1_score(testing_Y, predict_Y, average = 'weighted')
process_time = finish_time - start_time

tick_labels = ["Trojan", "Downloader", "Miner", "Ransom"]
ax = sn.heatmap(cm / np.sum(cm), annot = True, fmt = '.2%', 
           xticklabels = tick_labels, yticklabels = tick_labels, cmap = 'Blues')
ax.set_title('Confusion Matrix for Naive Bayes classifier');
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
plt.show()

print("Accuracy score: " + str(acc * 100) + "%")
print("Precision score: " + str(pre * 100) + "%")
print("Recall score: " + str(rec * 100) + "%")
print("f-measure score: " + str(f_mea * 100) + "%")
print("Processing time: " + str(process_time) + " seconds")


# Random Forest - Multiclass Classification

start_time = time.time()

RF_model = RandomForestClassifier(n_estimators = 13)

RF_model.fit(training_X_std, training_Y)

predict_Y = RF_model.predict(testing_X_std)

finish_time = time.time()

cm = confusion_matrix(testing_Y, predict_Y)
acc = accuracy_score(testing_Y, predict_Y)
pre = precision_score(testing_Y, predict_Y, average = 'weighted')
rec = recall_score(testing_Y, predict_Y, average = 'weighted')
f_mea = f1_score(testing_Y, predict_Y, average = 'weighted')
process_time = finish_time - start_time

tick_labels = ["Trojan", "Downloader", "Miner", "Ransom"]
ax = sn.heatmap(cm / np.sum(cm), annot = True, fmt = '.2%', 
           xticklabels = tick_labels, yticklabels = tick_labels, cmap = 'Blues')
ax.set_title('Confusion Matrix for Random Forest classifier');
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
plt.show()

print("Accuracy score: " + str(acc * 100) + "%")
print("Precision score: " + str(pre * 100) + "%")
print("Recall score: " + str(rec * 100) + "%")
print("f-measure score: " + str(f_mea * 100) + "%")
print("Processing time: " + str(process_time) + " seconds")


# k Nearest Neighbours - Multiclass Classification

start_time = time.time()

kNN_model = KNeighborsClassifier(n_neighbors = 4)

kNN_model.fit(training_X_std, training_Y)

predict_Y = kNN_model.predict(testing_X_std)

finish_time = time.time()

cm = confusion_matrix(testing_Y, predict_Y)
acc = accuracy_score(testing_Y, predict_Y)
pre = precision_score(testing_Y, predict_Y, average = 'weighted')
rec = recall_score(testing_Y, predict_Y, average = 'weighted')
f_mea = f1_score(testing_Y, predict_Y, average = 'weighted')
process_time = finish_time - start_time

tick_labels = ["Trojan", "Downloader", "Miner", "Ransom"]
ax = sn.heatmap(cm / np.sum(cm), annot = True, fmt = '.2%', 
           xticklabels = tick_labels, yticklabels = tick_labels, cmap = 'Blues')
ax.set_title('Confusion Matrix for KNN classifier');
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
plt.show()

print("Accuracy score: " + str(acc * 100) + "%")
print("Precision score: " + str(pre * 100) + "%")
print("Recall score: " + str(rec * 100) + "%")
print("f-measure score: " + str(f_mea * 100) + "%")
print("Processing time: " + str(process_time) + " seconds")


# Support Vector Machines - Multiclass Classification

start_time = time.time()

SVM_model = svm.SVC(kernel='poly')

SVM_model.fit(training_X_std, training_Y)

predict_Y = SVM_model.predict(testing_X_std)

finish_time = time.time()

cm = confusion_matrix(testing_Y, predict_Y)
acc = accuracy_score(testing_Y, predict_Y)
pre = precision_score(testing_Y, predict_Y, average = 'weighted')
rec = recall_score(testing_Y, predict_Y, average = 'weighted')
f_mea = f1_score(testing_Y, predict_Y, average = 'weighted')
process_time = finish_time - start_time

tick_labels = ["Trojan", "Downloader", "Miner", "Ransom"]
ax = sn.heatmap(cm / np.sum(cm), annot = True, fmt = '.2%', 
           xticklabels = tick_labels, yticklabels = tick_labels, cmap = 'Blues')
ax.set_title('Confusion Matrix for SVM classifier');
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
plt.show()

print("Accuracy score: " + str(acc * 100) + "%")
print("Precision score: " + str(pre * 100) + "%")
print("Recall score: " + str(rec * 100) + "%")
print("f-measure score: " + str(f_mea * 100) + "%")
print("Processing time: " + str(process_time) + " seconds")


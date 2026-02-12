from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from Question_4 import features_train, labels_train, features_test, labels_test

kValues = [1,3,5,7,9]
listofAccuracyResults = []

for k in kValues:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(features_train, labels_train)
    predictedLabels = knn_model.predict(features_test)
    accuracy = accuracy_score(labels_test, predictedLabels)
    listofAccuracyResults.append(accuracy)


resultsDataFrame = pd.DataFrame({"k": kValues,"Test Accuracy":listofAccuracyResults})

print(resultsDataFrame)

# Changing the value of k affects how sensitive the KNN model is to data points. As a result smaller k values will make the model more flexible whilst larger k values will make the model smoother
# Very small values of k can cause overfitting because the model relies on only a group of nearby points and very small values may lead to the model capturing these outliers as if they are important patterns, making it fail on new data
# Very large values of k may cause underfitting because the model averages over too much information leading it to miss important patterns
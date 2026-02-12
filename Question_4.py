from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score
import pandas as pd

kidney_disease_dataFrame = pd.read_csv("kidney_disease.csv")

featureMatrix = kidney_disease_dataFrame._get_numeric_data()
targetLabels = kidney_disease_dataFrame["classification"].map({"ckd":1,"notckd":0})

featureMatrix = featureMatrix.dropna()
targetLabels = targetLabels.loc[featureMatrix.index]

features_train, features_test, labels_train, labels_test = train_test_split(featureMatrix,targetLabels,test_size=0.3,random_state=42)

knnModel = KNeighborsClassifier(n_neighbors=5)
knnModel.fit(features_train,labels_train)

predictedLabels = knnModel.predict(features_test)

if __name__ == "__main__":
    print("Confusion Matrix:\n", confusion_matrix(labels_test,predictedLabels))
    print("Accuracy:", accuracy_score(labels_test,predictedLabels))
    print("Precision:", precision_score(labels_test,predictedLabels))
    print("Recall:", recall_score(labels_test,predictedLabels ))
    print("F1-score:", f1_score(labels_test,predictedLabels ))

# In kidney disease prediction, a True positive means that the model correctly predicts that a patient has kidney disease whilst a True negative means that the model correctly predicts that a patient is healthy.
#Conversely, a False positive means that the model predicts kidney disease for a healthy patient whilst a False negative means that the model fails to detect kidney disease in a patient that has it.
# Accuracy alone is not enough because a model can appear accurate but can be missing important disease cases, which may be a result of missing or imbalanced data
# The metric that is the most important in the case of missing a kidney disease that is very serious is recall, as this measures how well the model identifies patients who actually have kidney disease

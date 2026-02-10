import pandas as pd
from sklearn.model_selection import train_test_split

kidney_disease_dataFrame = pd.read_csv("kidney_disease.csv")


matrix_X = kidney_disease_dataFrame.drop(columns=["classification"])
vector_Y = kidney_disease_dataFrame["classification"]


matrix_X_train, matrix_X_test, vector_Y_train, vector_Y_test = train_test_split(matrix_X, vector_Y, test_size=0.30, random_state=42)

# We should not train and tets a model on the same data because the model may end up memorizing the training examples. This can potentially lead to unrealistic performance which will not reflect how the model will perform on new data
#The purpose of a testing set it to evaluate how well a trained model gets accustomed to unseen data and it also provides the user an estimate of the model's performance.
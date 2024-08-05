import sklearn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

import sklearn.metrics
import sklearn.model_selection
import sklearn.neighbors
from tool.create_confusion_matrix import CREATE_CONFUSION_MATRICS

raw_data_df  = pd.read_csv(os.path.join(r"D:\machine_learning_AI_Builders\ML_Algorithm\KNN_Pima_Indians_Diabetes_Database\data\diabetes.csv"))

print(raw_data_df.shape)
print(raw_data_df)

feature = raw_data_df.drop(labels="Outcome",axis=1)
targets = raw_data_df["Outcome"]

print(feature)
print(targets)

x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(feature,targets,test_size=0.2,shuffle=True,random_state=42)

print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)


y_train = y_train.values.reshape(-1,1)
y_test = y_test.values.reshape(-1,1)


print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)


# k_neighbors  = np.arange(1,10)

# for idx,k in enumerate(k_neighbors):

#     knn_model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k)
#     knn_model.fit(x_train,y_train)
#     eval_values = knn_model.score(x_test,y_test)
#     print(eval_values)

## k == 7 ##
knn_model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=7)
#knn_model = sklearn.neighbor.KNeighborsClassifier(n_neighbors=7)
knn_model.fit(x_train,y_train)

y_pred = knn_model.predict(x_test)


CREATE_CONFUSION_MATRICS(y_actual=y_test,y_pred=y_pred,numclass=len(set(targets))-1)

print(sklearn.metrics.classification_report(y_true=y_test,y_pred=y_pred))



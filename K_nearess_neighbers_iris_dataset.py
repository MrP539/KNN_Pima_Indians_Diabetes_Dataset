import sklearn
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.neighbors
import numpy as np

#################################################################  load and processing data  ##########################################################

iris_data = sklearn.datasets.load_iris()

print(iris_data.keys())

class_name = iris_data['target_names']


feature = iris_data["data"]
target = iris_data["target"]

print(feature.shape,target.shape)

x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(feature,target,test_size=0.2,random_state=101,shuffle=True)

print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

print(x_train[0],y_test[0])
##################################################################  build model and train  ##############################################################

knn_model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)

knn_model.fit(X=x_train,y=y_train)

###################################################################  prediction  ########################################################################''

pred = knn_model.predict([x_test[0]])

accuracy = knn_model.score(X=x_test,y=y_test)

print(f"Accuracy : {float(accuracy)*100}")


print(f"Prediction : {class_name[pred.item()]} Vs Labels : {class_name[y_test[0].item()]}")

print(sklearn.metrics.classification_report(y_pred=knn_model.predict(x_test),y_true=y_test,target_names=class_name))


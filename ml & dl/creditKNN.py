import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score

# đọc file
credit_data = pd.read_csv("credit_data.csv")
# print(credit_data)

features = credit_data[["income", "age", "loan"]]
target = credit_data["default"]
X = np.array(features).reshape([-1, 3])
Y = np.array(target)


X = preprocessing.MinMaxScaler().fit_transform(X)
features_train, features_test, target_train, target_test = train_test_split(
    X, Y, test_size=0.3)
model = KNeighborsClassifier(n_neighbors=32)
model_KNN = model.fit(features_train, target_train)

# kiểm tra số k để đạt độ chính xác đạt max
# b = []
# for k in range(1, 100):
#     knn = KNeighborsClassifier(n_neighbors=k)
#     score = cross_val_score(knn, X, Y, cv=10, scoring="accuracy")
#     b.append(score.mean())
# print(np.argmax(b))


predictions = model_KNN.predict(features_test)
print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test, predictions))

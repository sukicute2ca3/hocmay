import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
# đọc file csv
credit_data = pd.read_csv("credit_data.csv")
# # hàm mũ
# print(credit_data.head())
# #hàm mô tả
# print(credit_data.describe())
# # hàm tương quan
# print(credit_data.corr())

features = credit_data[["income", "age", "loan"]]
target = credit_data["default"]
print(features)
# 30% của dữ liệu là thử nghiêm,70% là đào tạo
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.3)
# 70% dữ liệu để tìm b0,b1,b2,b3 trong quy hồi logistic x1 là income x2 là age x3 là loan
# khởi tạo hồi quy tuyến tính dưới dạng mô hình
model = LogisticRegression()
model.fit = model.fit(features_train, target_train)
# in ra giá trị b0
print(model.fit.intercept_)
# in ra các giá trị b1,b2,b3
print(model.fit.coef_)
predictions = model.fit.predict(features_test)
print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test, predictions))

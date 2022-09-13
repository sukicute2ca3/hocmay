import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

credit_data = pd.read_csv("credit_data.csv")
feature = credit_data[["income", "age", "loan"]]
target = credit_data["default"]

# máy học xử lí mảng chứ không phải khung dữ liệu
x = np.array(feature).reshape((-1, 3))
y = np.array(target)


# tạo mô hình hồi quy logistic
model = LogisticRegression()

# sử dụng xác thực chéo trên mô hình
predicted = cross_validate(model, x, y, cv=10)
print(np.mean(predicted["test_score"]))

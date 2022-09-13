import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math
# đọc file cvs bằng pandas
house_data = pd.read_csv('house_prices.csv')
# print(house_data)

size = house_data["sqft_living"]
price = house_data["price"]
# máy học xử lí mảng chứ không phải khung dữ liệu

x = np.array(size).reshape(-1, 1)
y = np.array(price).reshape(-1, 1)
print(size)
print(x)
print(y)
z = np.zeros((10, 1))


# sử dụng hồi quy tuyến tính và hàm phù hợp để đào tạo mô hình đã cho
model = LinearRegression()
model.fit(x, y)

# sai số bình phương trung bình và giá trị R
regression_model_mse = mean_squared_error(x, y)
print("MSE : ", math.sqrt(regression_model_mse))
print("R squared value : ", model.score(x, y))

# ta nhận được các  trị B sau khi mô hình f hợp
# đây là b0 của mô hình
print("b0 : ", model.coef_[0])
# đây là b1
print("b1 : ", model.intercept_[0])


plt.scatter(x, y, color="green")
plt.plot(x, model.predict(x), color="black")
plt.title("Linear Regression")
plt.xlabel("Size")
plt.ylabel("Price")
plt.ylabel("ham 0")
plt.show()

# gái được tính bởi mô hình
print("Price by the model : ", model.predict([[2000]]))

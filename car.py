# importing required libraries
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import cross_val_score, train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor ,GradientBoostingRegressor,BaggingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score
import pickle
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.datasets import load_iris
# Loading Dataset
df = pd.read_csv('car.csv')
x = df.drop('Price',axis=1)
y = df['Price']
scaler = StandardScaler()
scaler.fit(x[['Mileage','EngineV']])
inputs_scaled = scaler.transform(x[['Mileage','EngineV']])
scaled_data = pd.DataFrame(inputs_scaled,columns=['Mileage','EngineV'])
x =scaled_data.join(x.drop(['Mileage','EngineV'],axis=1))

print(x.head())
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
models = [ LinearRegression, DecisionTreeRegressor, RandomForestRegressor, Ridge]
mse = []
rmse = []
evs = []
r_square_score = []

for model in models:
    regressor = model().fit(X_train, y_train)
    pred = regressor.predict(X_test)
    mse.append(mean_squared_error(y_true= y_test, y_pred= pred))
    rmse.append(np.sqrt(mean_squared_error(y_true= y_test, y_pred= pred)))
    evs.append(explained_variance_score(y_true= y_test, y_pred= pred))
    r_square_score.append(r2_score(y_true= y_test, y_pred= pred))

MLModels_df = pd.DataFrame({"Models": [ 'Linear Regression', 'Decision Tree Regressor', 'Random Forest Regressor', 'Ridge'],
                           "Mean Squared Error": mse,
                           "Root Mean Squared Error": rmse,
                           "Explained Variance Score": evs,
                           "R-Square Score ": r_square_score,})

MLModels_df.set_index('Models', inplace=True)
print(MLModels_df.head())

regressor = RandomForestRegressor()
# Số cây
n_estimators = [int(x) for x in np.linspace(start=40, stop=500, num=15)]
# Số đặc trưng
max_features = ['auto', 'sqrt']
# Chiều sâu tối đa
max_depth = [int(x) for x in np.linspace(start= 5, stop= 30, num= 6)]
# Số mẫu yêu cầu để chia 
min_samples_split = [2,5,10,15]
# số lượng điểm dữ liệu tối thiểu cần để một nút lá (nút cuối cùng của cây) có thể tồn tại
min_samples_leaf = [1,2,5]
# Tạo the random grid
random_grid= {'n_estimators': n_estimators, 
              'max_features' : max_features,
              'max_depth' : max_depth,
              'min_samples_split' : min_samples_split,
              'min_samples_leaf' : min_samples_leaf}
#print(random_grid)
regressor_random = RandomizedSearchCV(estimator=  regressor, param_distributions=  random_grid, scoring= 'neg_mean_squared_error', 
                                      n_iter = 25, cv=3, verbose = 2, n_jobs=1)
regressor_random.fit(X_train, y_train)
y_predictions = regressor_random.predict(X_test)
# Lưu mô hình vào tệp
#joblib.dump(regressor_random.best_estimator_, 'random.pkl')

print(regressor_random.best_estimator_)
print('Mean Squareed Error: ', mean_squared_error(y_test, y_predictions))
print('Root Mean Square Error: ', np.sqrt(mean_squared_error(y_test, y_predictions)))
print('R-Square Score : ', r2_score(y_test, y_predictions))
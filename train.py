import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
import pickle

df = pd.read_csv('USA_Housing.csv')
df.drop(columns = ['Address'], inplace = True)

X = df.drop(columns = ['Price'])
y = df['Price']


scaler = StandardScaler()
xcolumns = X.columns
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns = xcolumns)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 100)

lr = LinearRegression()
lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)
print('R2 Score:', r2_score(y_test,y_pred))
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("RMSE:",rmse)
with open('house_model.pkl', 'wb') as f:
    pickle.dump(lr,f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler,f)
print("✅Sucessfully : 'house_model.pkl'and 'scaler.pkl' created")
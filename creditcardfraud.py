import pandas as pd
import numpy as np
import matplotlib as plt

data=pd.read_csv(r'C:\Users\ANANYA RATURI\Desktop\creditcard.csv')
fraud = data[data['Class'] == 1] 
valid = data[data['Class'] == 0] 
outlierFraction = len(fraud)/float(len(valid)) 
print(outlierFraction) 
print('Fraud Cases: {}'.format(len(data[data['Class'] == 1]))) 
print('Valid Transactions:{}'.format(len(data[data['Class'] == 0]))) 
target='Class'
x=data.drop(['Class'],axis=1)
y=data[target]
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(x)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=101)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
clf=LinearRegression()
clf.fit(X_train,y_train)
Y_pred=clf.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(y_test, Y_pred))

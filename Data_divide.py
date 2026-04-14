# Machine Learning
# Data(50) --> x(input data),y(Target data)
# x ---> x_train, x_test
# y ---> y_train, y_test
# for Machine training we will pass (x_train,y_train)
# for Machine testing (x_test) ---> (y_test)

1+1 = 2
1+2 = 3
1+4 = 5
1+6 = 7
1+7 = 8
1+8 = 9

# x = (1+1,1+2,1+4,....1+8)
# y = (2,3,5,7,8,9)
# x_train = (1+1,1+2,1+4,1+6)
# x_test = (1+7,1+8)
# y_train = (2,3,5,7)
# y_test = (8,9)

import pandas as pd

df = pd.read_csv("/content/covid_toy - covid_toy.csv")
df.head(2)

df.shape

df.isnull().sum()

df['fever'] = df['fever'].fillna(df['fever'].mean())

df.isnull().sum()

x = df.drop(columns=['has_covid'])
y = df['has_covid']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

print("total Data Shape",df.shape)
print("------------------------------")
print("x Shape",x.shape)
print("--------------------------------")
print("x_train Shape",x_train.shape)
print("x_test Shape",x_test.shape)
print("--------------------------------")
print("y Shape",y.shape)
print("--------------------------------")
print("y_train Shape",y_train.shape)
print("y_test Shape",y_test.shape)

df1 = pd.read_csv("/content/placement - placement.csv")
df1.head(2)

df1.shape

x = df1.drop(columns=['placed'])
y = df1['placed']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

print("total Data Shape",df1.shape)
print("------------------------------")
print("x Shape",x.shape)
print("--------------------------------")
print("x_train Shape",x_train.shape)
print("x_test Shape",x_test.shape)
print("--------------------------------")
print("y Shape",y.shape)
print("--------------------------------")
print("y_train Shape",y_train.shape)
print("y_test Shape",y_test.shape)

import numpy as np
import pandas as pd

df = pd.read_csv('/content/covid_toy - covid_toy.csv')
df.head(2)

df.isnull().sum()

df['fever'] = df['fever'].fillna(df['fever'].mean())

x = df.drop(columns=['city'])
x

x['gender'] = x['gender'].map({'Male':1,'Female':0})

x['cough'] = x['cough'].map({'Mild':1,'Strong':2})

x['has_covid'] = x['has_covid'].map({'Yes':1,'No':0})

x

from sklearn.model_selection import train_test_split

x = x.drop(columns=['has_covid'])
y = df['has_covid']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

np.round(x_train.describe(),2)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)   # fit means learn the parameters and transform means apply on that

x_train_new = pd.DataFrame(x_train,columns=x.columns)

type(x_train)

type(x_train_new)

np.round(x_train_new.describe(),2)

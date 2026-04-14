import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Encoding ---> transform your categorical data into numerical data

# (1). Labelencoder --> has_covid['yes','no']  ---> encode --> has_covid[1,0]

df = pd.read_csv('/content/covid_toy - covid_toy.csv')

df = df.dropna()

from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()

df['gender'] = lb.fit_transform(df['gender'])

df['cough'] = lb.fit_transform(df['cough'])

df['city'] = lb.fit_transform(df['city'])
df['has_covid'] = lb.fit_transform(df['has_covid'])

df.head()

x = df.drop(columns=['has_covid'])
y = df['has_covid']

x_train, y_train, x_test, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# applying standardization for converting my data into normal distribution

sc = StandardScaler()

x_train_sc = sc.fit_transform(x_train)

x_train_new = pd.DataFrame(x_train_sc,columns=x_train.columns)

np.round(x_train_new.describe(),2)

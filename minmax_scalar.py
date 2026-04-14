# normalization ---> using minmaxscaler
# data min_value = 0 and max_value = 1

# play store ---> different_types_app --> 1 app download 4.2k, 1 app downloads 4.3M
# 1 app download 650
# in this condition, if I apply any operation, then it will return

import numpy as np
import pandas as pd

df = pd.read_csv("/content/covid_toy - covid_toy.csv")

df = df.dropna()

df = df.drop(columns = ['city'])

# step-1  ::: convert categorical data into numerical data

df['gender'] = df['gender'].map({"Female":1,"Male":0})

df['cough'] = df['cough'].map({"Mild":0,"Strong":1})

df['has_covid'] = df['has_covid'].map({"Yes":1,"No":0})

# step-2 data dividation

x = df.drop(columns = ['has_covid'])
y = df['has_covid']

from sklearn.model_selection import train_test_split

x_train, y_train, x_test, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

np.round(x_train.describe(),2)

# step-3 applying minmaxscaler

from sklearn.preprocessing import MinMaxScaler

mn = MinMaxScaler()

x_train_mn = mn.fit_transform(x_train)

x_train_new = pd.DataFrame(x_train_mn,columns = x_train.columns)

np.round(x_train_new.describe(),2)

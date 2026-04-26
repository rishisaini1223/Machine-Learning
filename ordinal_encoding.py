# LabelEncode ---> when we have single column. we can't sub-category form
# has_covid['Yes','No']  --> labelencoder ---> [1,0]
# ordinalencoder ---> when we have multiple column. we can set sub-category form
# has_covid['Yes','No']   --> ordinal encoder ---> [0,1]

# first approach :::::
# df ---> categorical column ---> ordinal encoder ---> data transform
# second approach :::
# df ---> x,y --->
# x ---> encode by ordinal encoder
# y ---> encode by label encoder


import numpy as np
import pandas as pd

# ordinal encoding --->
# label encoding ---> column[sub1,sub2,sub3,sub4,sub5]  ---> encode
# ordinal encoding ----> column[sub1,sub2,sub3]  ---> Rearrange ---> encode

df = pd.read_csv("/content/covid_toy - covid_toy.csv")
df.head(2)

df.dropna()

df = df.drop(columns=['city','age','fever'])
df.head(2)

from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder(categories=[['Male','Female'],['Mild','Strong'],['No','Yes']])

oe_ar = oe.fit_transform(df)

new_df = pd.DataFrame(oe_ar,columns=df.columns)

new_df.head(2)

#  Second way 

df = pd.read_csv("/content/covid_toy - covid_toy.csv")
df.head(2)

df = df.drop(columns=['city','age','fever'])
df.head(2)

df = df.dropna()

df.head(2)

x = df.drop(columns=['has_covid'])
y = df['has_covid']

from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder(categories=[['Male','Female'],['Mild','Strong']])

oe_ar_in = oe.fit_transform(x)
# oe_ar_in

y.head(2)

from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()

y = lb.fit_transform(y)

y


y = lb.y = lb.fit_transyform(y)

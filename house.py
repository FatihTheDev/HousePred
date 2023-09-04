import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('housing.csv')

data.dropna(inplace=True)

data.drop(['ocean_proximity'], axis=1, inplace=True)


from sklearn.model_selection import train_test_split

x = data.drop(['median_house_value'], axis=1)
y = data['median_house_value']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

train_data = x_train.join(y_train)

plt.figure(figsize=(15,8))
sns.heatmap(train_data.corr(), annot=True)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train, y_train)

print(model.score(x_test, y_test))
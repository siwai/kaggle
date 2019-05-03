import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# train data load(csvFileのPath気をつけてください)
train = pd.read_csv("train.csv", index_col="PassengerId")
test = pd.read_csv("train.csv", index_col="PassengerId")
# 出力確認
# print(train.shape)
# Macの場合 ctrl + Enterで実装 or Runボタンをクリック
# print(train.head())
low_fare = train[train["Fare"] < 100]

# print(train.head)
# sns.countplot(data=train, x="Sex", hue="Survived")

# 年齢と料金の関連性を確認
# sns.lmplot(data=train, x="Age", y="Fare", hue="Survived", fit_reg=False)
sns.lmplot(data=low_fare, x="Age", y="Fare", hue="Survived", fit_reg=False)

# 凡例の表示
plt.legend()

# プロット表示(設定の反映)
plt.show()
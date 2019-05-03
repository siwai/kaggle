import pandas as pd
# train data load(csvFileのPath気をつけてください)
train = pd.read_csv("train.csv", index_col = "PassengerId")
test = pd.read_csv("train.csv", index_col = "PassengerId")
# 出力確認
print(train.shape)
# Macの場合 ctrl + Enterで実装 or Runボタンをクリック
# print(train.head())

# print(train.head)
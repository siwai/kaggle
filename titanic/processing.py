import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# data
train = pd.read_csv("train.csv", index_col="PassengerId")
test = pd.read_csv("train.csv", index_col="PassengerId")

# Sex_encodeというカラムに性別を数字で格納
train.loc[train["Sex"] == "male", "Sex_encode"] = 0
train.loc[train["Sex"] == "female", "Sex_encode"] = 1

# testも同様
test.loc[train["Sex"] == "male", "Sex_encode"] = 0
test.loc[train["Sex"] == "female", "Sex_encode"] = 1

# NaNに値を埋めるために、Fare_fillinカラムを作成
train["Fare_fillin"] = train["Fare"]

#test.csvにも適用する
test["Fare_fillin"] = test["Fare"]

# FareがNaNになっている乗客を検索後、0を入れる
test.loc[test["Fare"].isnull(), "Fare_fillin"] = 0

# 0が入っているか確認
# test.loc[test["Fare"].isnull(), ["Fare", "Fare_fillin"]]

#Age 20歳以下を学生とする
train["Student"] = train["Age"] < 20

#test.csvにも適用する
test["Student"] = test["Age"] < 20

label_name = "Survived"

feature_names = [ "Sex_encode", "Fare_fillin", "Student"]
X_train = train[feature_names]

#test.csvにも適用する
X_test = test[feature_names]
y_train = train[label_name]

# scikit-learn(sklearn)の tree ModuleからDecisionTreeClassifierを取得
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=8, random_state=0)
# DecisionTreeClassifierを学習
# fit という機能でtrainデータのfeature(X_train)とlabel(y_train)を入れる
model.fit(X_train, y_train)
# fitが終わったら、predictの機能でSurvivedを予測する
#　その後、testデータのSurvivedを返して、predictionsという変数に格納
predictions = model.predict(X_test)

# Kaggleが提供している提出向けcsvを読み取る
# PassengerIdは testデータと一緒で, Survivedはmale(0), female(1)が入っている
submission = pd.read_csv("gender_submission.csv", index_col="PassengerId")
print(submission.shape)
print(submission.head())

# kaggleに提出するcsv作成
submission.to_csv("submit_199503.csv")
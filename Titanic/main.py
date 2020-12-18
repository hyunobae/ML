import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

def showhist():#시각화
    train.hist(bins=50, figsize=(20,15))
    plt.show()

def showheatmap():
    plt.figure(figsize=(9, 9))
    sns.heatmap(train.corr(), linewidths=0.1, vmax=0.5,
                linecolor='white', annot=True)
    plt.show()  # fare와 생존의 상관관계가 높았다. 1이 생존한 경우


train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")
submission = pd.read_csv("./gender_submission.csv")

#showhist()
#showheatmap()

# print(train.isnull().sum()) #age 177개 , cabin 687개, embarked 2 비어있는 데이터 있다 -> 채워줘야 분석 가능
# print(test.isnull().sum()) #age개 86개, fare 1개, cabin 327개 비어있다. 채우자.

train['Cabin'].replace('-', np.nan, inplace=True)
test['Cabin'].replace('-', np.nan, inplace=True)

#나이는 중간값으로 채워보자
train.fillna(train['Age'].mean(), inplace=True)
#cabin은 어떻게 처리할래?
train.fillna(train['Embarked'].mode(), inplace=True)

test.fillna(test['Age'].mean(), inplace=True)
test.fillna(test['Fare'].median(), inplace=True)
test.fillna(test['Cabin'].mode(), inplace=True)

train['Cabin'] = train['Cabin'].str[:1]
print(train['Cabin']) #문자열 슬라이싱했고, NaN값들 채워넣어야 함 이떄 mode써보자
train.fillna(train['Cabin'].value_counts(dropna=True).idxmax(), inplace=True)

test['Cabin'] = test['Cabin'].str[:1]
test.fillna(test['Cabin'].value_counts(dropna=True).idxmax(), inplace=True)

#print(train.isnull().sum())
#print(test.isnull().sum())# 비어있는 데이터 없다.

#직접적인 영향을 주지 않는 데이터는 삭제해보자
train.drop(['Name'], axis=1, inplace=True) # 이름의 경우 이미 fare로 사회적 지위가 반영되어서 지워봄
test.drop(['Name'], axis=1, inplace=True)
train.drop(['Ticket'], axis=1, inplace=True) # 표 번호는 의미없을것 같다..
test.drop(['Ticket'], axis=1, inplace=True)
train.drop(['Embarked'], axis=1, inplace=True) #필요없다
test.drop(['Embarked'], axis=1, inplace=True)

#시각화 하기위해 cabin도 encoding
cabin = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F':5, 'G':6, 'H':7, 'I':8, 'J':9} #히트맵으로 시각화하기 위해 캐빈도 맵핑해본다.
train['Cabin'] = train['Cabin'].map(cabin)
test['Cabin'] = test['Cabin'].map(cabin)
#성별도 인코딩
gender = {'male': 0, 'female': 1}
train['Sex'] = train['Sex'].map(gender)
test['Sex'] = test['Sex'].map(gender)
#print(train['Sex'].head())

#print(train.head())
#showheatmap()#성별과 생존의 관계가 0.54로 가장 높았다.

#딱 1개의 cabin이 전처리 과정을 진행해도 NaN이어서 drop
train.dropna(subset=['Cabin'], how='any', axis=0, inplace=True)

print(train.isnull().sum())
print(test.isnull().sum())

x_train = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch','Fare','Cabin']]
y_train = train['Survived']

rf_clf = RandomForestClassifier(max_depth=50, n_estimators=40)
rf_clf.fit(x_train, y_train)

x_val = test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch','Fare','Cabin']]
y_val = rf_clf.predict(x_val)

print('training set accuracy:', rf_clf.score(x_train, y_train))
print('validation set accuracy:', rf_clf.score(x_val, y_val))

submit = pd.DataFrame({
    'PassengerId' : test['PassengerId'],
    'Survived' : y_val
})
submit.to_csv('save.csv', index=False)
#dt가 0.76555, rf가 0.74641.. 전처리에서 개선을 해야할 듯..

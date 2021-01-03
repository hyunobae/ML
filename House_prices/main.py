import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")
result = pd.read_csv("./sample_submission.csv")

print(train.head())
print(train.shape)
print(test.shape)

# 결측값 확인
print(train.isnull().sum())  # LotFrontage 259개
print(test.isnull().sum())  # LotFrontage 227개, SaleType 1개, MSZoning 4개

train.info()
test.info()


# GrLivArea / SalePirce간 산점도 확인
# var = 'GrLivArea'
# data = pd.concat([train['SalePrice'], train[var]], axis=1)
# data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
# plt.show()

# 극단적인 데이터 삭제
# train = train[(train['GrLivArea'] < 4000) | (train['SalePrice'] > 700000)]
# data = pd.concat([train['SalePrice'], train[var]], axis=1)
# data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
# plt.show()

# 비어있는 데이터 정의따라 none, 평균값으로 주자
def nullToNone(df):
    df.loc[:, 'LotFrontage'] = df.loc[:, 'LotFrontage'].fillna(train['LotFrontage'].mean())
    df.loc[:, 'Alley'] = df.loc[:'Alley'].fillna('No')
    df.loc[:, 'MasVnrType'] = df.loc[:'MasVnrType'].fillna('None')
    df.loc[:, 'MasVnrArea'] = df.loc[:'MasVnrArea'].fillna(0)
    df.loc[:, 'BsmtQual'] = df.loc[:'BsmtQual'].fillna('No')
    df.loc[:, 'BsmtCond'] = df.loc[:'BsmtCond'].fillna('No')
    df.loc[:, 'BsmtExposure'] = df.loc[:, 'BsmtExposure'].fillna('No')
    df.loc[:, 'BsmtFinType1'] = df.loc[:, 'BsmtFinType1'].fillna('No')
    df.loc[:, 'BsmtFinType2'] = df.loc[:, 'BsmtFinType2'].fillna('No')
    df.loc[:, 'BsmtFullBath'] = df.loc[:, 'BsmtFullBath'].fillna(0)
    df.loc[:, 'BsmtHalfBath'] = df.loc[:, 'BsmtHalfBath'].fillna(0)
    df.loc[:, 'BsmtUnfSF'] = df.loc[:, 'BsmtUnfSF'].fillna(0)
    df.loc[:, 'FireplaceQu'] = df.loc[:, 'FireplaceQu'].fillna('No')
    df.loc[:, 'GarageType'] = df.loc[:, 'GarageType'].fillna('No')
    df.loc[:, 'GarageFinish'] = df.loc[:, 'GarageFinish'].fillna('No')
    df.loc[:, 'GarageQual'] = df.loc[:, 'GarageQual'].fillna('No')
    df.loc[:, 'GarageCars'] = df.loc[:, 'GarageCars'].fillna(0)
    df.loc[:, 'GarageArea'] = df.loc[:, 'GarageArea'].fillna(0)
    df.loc[:, 'GarageCond'] = df.loc[:, 'GarageCond'].fillna('No')
    df.loc[:, 'PoolQC'] = df.loc[:, 'PoolQC'].fillna('No')
    df.loc[:, 'Fence'] = df.loc[:, 'Fence'].fillna('No')
    df.loc[:, 'Utilities'] = df.loc[:, 'Utilities'].fillna('AllPub')
    df.loc[:, 'Exterior1st'] = df.loc[:, 'Exterior1st'].fillna(900)
    df.loc[:, 'Exterior2nd'] = df.loc[:, 'Exterior2nd'].fillna(0)
    df.loc[:, 'MiscFeature'] = df.loc[:, 'MiscFeature'].fillna('No')
    df.loc[:, 'TotRmsAbvGrd'] = df.loc[:, 'TotRmsAbvGrd'].fillna(0)
    df.loc[:, 'MiscVal'] = df.loc[:, 'MiscVal'].fillna(0)
    df.loc[:, 'OpenPorchSF'] = df.loc[:, 'OpenPorchSF'].fillna(0)
    df.loc[:, 'PavedDrive'] = df.loc[:, 'PavedDrive'].fillna('N')
    df.loc[:, 'LotShape'] = df.loc[:, 'LotShape'].fillna('Reg')
    df.loc[:, 'KitchenQual'] = df.loc[:, 'KitchenQual'].fillna('TA')
    df.loc[:, 'SaleCondition'] = df.loc[:, 'SaleCondition'].fillna('Normal')
    df.loc[:, 'SaleType'] = df.loc[:, 'SaleType'].fillna('WD')
    df.loc[:, 'MSZoning'] = df.loc[:, 'MSZoning'].fillna('RL')
    df.loc[:, 'Electrical'] = df.loc[:, 'Electrical'].fillna('SBrkr')
    df.loc[:, 'TotalBsmtSF'] = df.loc[:, 'TotalBsmtSF'].fillna(900)
    return df


train = nullToNone(train)
test = nullToNone(test)


def mapping(df):  # 숫자가 유의미한 의미를 나타내는 경우 변환
    df = df.replace({"MSSubClass": {20: "SC20", 30: "SC30", 40: "SC40", 45: "SC45",
                                    50: "SC50", 60: "SC60", 70: "SC70", 75: "SC75",
                                    80: "SC80", 85: "SC85", 90: "SC90", 120: "SC120",
                                    150: "SC150", 160: "SC160", 180: "SC180", 190: "SC190"},
                     "MoSold": {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                                7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"},
                     })

    df = df.replace({"Alley": {"Grvl": 1, "Pave": 2},
                     "BsmtCond": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                     "BsmtExposure": {"No": 0, "Mn": 1, "Av": 2, "Gd": 3},
                     "BsmtFinType1": {"No": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
                     "BsmtFinType2": {"No": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
                     "BsmtQual": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                     "ExterCond": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                     "ExterQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                     "FireplaceQu": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                     "Functional": {"Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5,
                                    "Min2": 6, "Min1": 7, "Typ": 8},
                     "GarageCond": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                     "GarageQual": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                     "HeatingQC": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                     "KitchenQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                     "LandSlope": {"Sev": 1, "Mod": 2, "Gtl": 3},
                     "LotShape": {"IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4},
                     "PavedDrive": {"N": 0, "P": 1, "Y": 2},
                     "PoolQC": {"No": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
                     "Street": {"Grvl": 1, "Pave": 2},
                     "Utilities": {"ELO": 1, "NoSeWa": 2, "NoSewr": 3, "AllPub": 4}}
                    )
    return df


train = mapping(train)
test = mapping(test)

# print(train.corr())

def drawplt():
    plt.figure(figsize=(25, 25))
    sns.heatmap(train.corr(), linewidths=0.1, vmax=0.5,
                linecolor='white', annot=True)
    plt.show()  #Lotshape가 연관성이 많이 떨어져보임 / EnclosedPorch도 낮아보인다 -> 지워도 되겠다.

# drawplt()

train.drop(['LotShape'], axis=1, inplace=True)
test.drop(['LotShape'], axis=1, inplace=True)
train.drop(['EnclosedPorch'], axis=1, inplace=True)
test.drop(['EnclosedPorch'], axis=1, inplace=True)

#0.5이상의 상관관계를 가진 특성만 추출
tcorr = train.corr()
tcorr.sort_values(['SalePrice'], ascending=False, inplace=True)
print(tcorr['SalePrice'])
print('type: ', type(tcorr))

#최종 결측값 확인
train[['OverallQual', 'GrLivArea', 'ExterQual', 'KitchenQual', 'GarageCars', 'GarageArea',
           'TotalBsmtSF', '1stFlrSF', 'BsmtQual', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'FireplaceQu', 'YearRemodAdd']].info()
test[['OverallQual', 'GrLivArea', 'ExterQual', 'KitchenQual', 'GarageCars', 'GarageArea',
           'TotalBsmtSF', '1stFlrSF', 'BsmtQual', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'FireplaceQu', 'YearRemodAdd']].info()

train_sort = list(tcorr[tcorr.SalePrice > 0.5][['SalePrice']].index.values)
train_sort.remove('SalePrice')
print(train_sort)

x_train = train[['OverallQual', 'GrLivArea', 'ExterQual', 'KitchenQual', 'GarageCars', 'GarageArea',
                'TotalBsmtSF', '1stFlrSF', 'BsmtQual', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'FireplaceQu', 'YearRemodAdd']]

trainList = list(train)
testList = list(test)

y_train = train.loc[:, 'SalePrice']
print(y_train)

x_test = test[['OverallQual', 'GrLivArea', 'ExterQual', 'KitchenQual', 'GarageCars', 'GarageArea',
           'TotalBsmtSF', '1stFlrSF', 'BsmtQual', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'FireplaceQu', 'YearRemodAdd']]


# print(train.shape)
# print(test.shape)

lr = LinearRegression()
lr.fit(x_train, y_train)

y_test = lr.predict(x_test)

print('training set accuracy:', lr.score(x_train, y_train))
print('validation set accuracy:', lr.score(x_test, y_test))

submit = pd.DataFrame({
    'Id' : test['Id'],
    'SalePrice' : y_test
})
submit.to_csv('save.csv', index=False)
#선형회귀 정확도 많이 떨어짐 -> 다른 방법 시도해보자 / 전처리가 부족한 이유일지도?
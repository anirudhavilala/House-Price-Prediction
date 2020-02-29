import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.tree import DecisionTreeRegressor
from copy import deepcopy as dp
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn import ensemble
import warnings
warnings.filterwarnings("ignore")
### input
traindt=pd.read_csv('train.csv')
testdt=pd.read_csv('test.csv')
Ydf = pd.DataFrame(traindt['SalePrice'])
Xdf = pd.concat([traindt.drop(['Id','SalePrice'],axis=1),testdt.drop(['Id'],axis=1)])
X1df=dp(Xdf)
X1df['MSSubClass']=X1df['MSSubClass'].astype('object')

### PoolQC Nan means there is no pool###
X1df['PoolQC']=X1df['PoolQC'].fillna(value='None')
### MiscFeature Nan implies None, i.e., there are no features that are not covered in other categories
X1df['MiscFeature']=X1df['MiscFeature'].fillna(value='None')
### Alley Nan implies no alley access
X1df['Alley']=X1df['Alley'].fillna(value='None')
### Fence
X1df['Fence']=X1df['Fence'].fillna(value='None')
### FireplaceQu Nan implies there is no fireplace
X1df['FireplaceQu']=X1df['FireplaceQu'].fillna(value='None')
### MasVnrType Nan implies there is no fireplace
X1df['MasVnrType']=X1df['MasVnrType'].fillna(value='None')
#X1df.columns.get_loc('MasVnrType') One of the values in area ins present as na in type
X1df.iloc[2610,24]=np.nan
### MasVnrArea Nan implies there is no fireplace
X1df['MasVnrArea']=X1df['MasVnrArea'].fillna(value=0)
### GarageCond Nan implies that there is no garage
X1df['GarageCond']=X1df['GarageCond'].fillna(value='None')
### GarageQual Nan implies that there is no garage
X1df['GarageQual']=X1df['GarageQual'].fillna(value='None')
### GarageFinish Nan implies that there is no garage
X1df['GarageFinish']=X1df['GarageFinish'].fillna(value='None')
### GarageType Nan implies that there is no garage
X1df['GarageType']=X1df['GarageType'].fillna(value='None')
### replacing a few true missing values
X1df.iloc[2126,63]=np.nan
X1df.iloc[2126,62]=np.nan
X1df.iloc[2126,59]=np.nan
X1df.iloc[2576,60]=0
X1df.iloc[2576,61]=0
X1df.iloc[2576,57]='None'

### BsmtExposure Nan implies there is no Basement
X1df['BsmtExposure']=X1df['BsmtExposure'].fillna(value='None')
### BsmtCond Nan implies there is no Basement
X1df['BsmtCond']=X1df['BsmtCond'].fillna(value='None')
### BsmtQual Nan implies there is no Basement
X1df['BsmtQual']=X1df['BsmtQual'].fillna(value='None')
### BsmtFinType2 Nan implies there is no Basement
X1df['BsmtFinType2']=X1df['BsmtFinType2'].fillna(value='None')
### BsmtFinType1 Nan implies there is no Basement
X1df['BsmtFinType1']=X1df['BsmtFinType1'].fillna(value='None')

#Removing columns that have high correlation with other columns
higcorr=['GarageYrBlt']
X1df=X1df.drop(higcorr,axis=1)

### Checking skewness of the feature variables and normalizing
### highly skewed features
skewness=X1df.skew(axis=0,skipna=True)
skewness=skewness[abs(skewness)>2]
skewed_features = list(skewness.index)
lam = 0.15
for feat in skewed_features:
    X1df[feat] = np.log1p(X1df[feat])
Ydf['SalePrice']=np.log1p(Ydf['SalePrice'])

### getting numeric and categorical attributes
numf=list(X1df.select_dtypes(include=[np.number]))
catf=list(X1df.select_dtypes(include=[np.object]))

X2df=dp(X1df)

#### replace numeric nans with median
for i in numf:
    X2df[i].fillna((X2df[i].median()),inplace=True)

#### replace categorical nans with mode
for j in catf:
    X2df[j].fillna((X2df[j].mode()[0]),inplace=True)

X3df=pd.concat([X2df[0:1460].drop(traindt[(traindt['GrLivArea']>4000) & (traindt['SalePrice']<300000)].index),X2df[1460:]])

#### Outlier detection and Handling ####
Ydf=Ydf.drop(traindt[(traindt['GrLivArea']>4000) & (traindt['SalePrice']<300000)].index)

#### Label Encoding to convert categorical data to numeric data

X3df['ExterQual']=X3df['ExterQual'].map({'Ex': 5, 'Fa': 2, 'Gd': 4, 'TA': 3, 'Po': 1})
X3df['ExterCond']=X3df['ExterCond'].map({'Ex': 5, 'Fa': 2, 'Gd': 4, 'Po': 1, 'TA': 3})
X3df['BsmtQual']=X3df['BsmtQual'].map({'Ex': 5, 'Fa': 2, 'Gd': 4, 'None': 0, 'TA': 3, 'Po':1})
X3df['BsmtCond']=X3df['BsmtCond'].map({'Ex': 5, 'Fa': 2, 'Gd': 4, 'None': 0, 'Po': 1, 'TA': 3})
X3df['BsmtExposure']=X3df['BsmtExposure'].map({'Av': 3, 'Gd': 4, 'Mn': 2, 'No': 1, 'None': 0})
X3df['BsmtFinType1']=X3df['BsmtFinType1'].map({'ALQ': 5, 'BLQ': 4, 'GLQ': 6, 'LwQ': 2, 'None': 0, 'Rec': 3, 'Unf': 1})
X3df['BsmtFinType2']=X3df['BsmtFinType2'].map({'ALQ': 5, 'BLQ': 4, 'GLQ': 6, 'LwQ': 2, 'None': 0, 'Rec': 3, 'Unf': 1})
X3df['HeatingQC']=X3df['HeatingQC'].map({'Ex': 5, 'Fa': 2, 'Gd': 4, 'Po': 1, 'TA': 3})
X3df['KitchenQual']=X3df['KitchenQual'].map({'Ex': 5, 'Fa': 2, 'Gd': 4, 'Po': 1, 'TA': 3})
X3df['FireplaceQu']=X3df['FireplaceQu'].map({'Ex': 5, 'Fa': 2, 'Gd': 4, 'None': 0, 'Po': 1, 'TA': 3})
X3df['GarageQual']=X3df['GarageQual'].map({'Ex': 5, 'Fa': 2, 'Gd': 4, 'None': 0, 'Po': 1, 'TA': 3})
X3df['GarageCond']=X3df['GarageCond'].map({'Ex': 5, 'Fa': 2, 'Gd': 4, 'None': 0, 'Po': 1, 'TA': 3})
X3df['PoolQC']=X3df['PoolQC'].map({'Ex': 4, 'Fa': 1, 'Gd': 3, 'None': 0, 'TA':2})
X3df['GarageFinish']=X3df['GarageFinish'].map({'Fin': 3, 'None': 0, 'RFn': 2, 'Unf': 1})
X3df['PavedDrive']=X3df['PavedDrive'].map({'N': 1, 'P': 2, 'Y': 3})

#### Removing redundant feature
X3df.drop(['Utilities'],axis=1)

#### Onehot Encoding
X3df=pd.get_dummies(X3df)

X4df=dp(X3df)
r2=[]
rmse=[]
X=X4df.iloc[0:Ydf.shape[0]]
X_test=X4df.iloc[Ydf.shape[0]:]
x_train,x_test,y_train,y_test=train_test_split(X,Ydf,test_size=0.3,random_state=0)

##### Decision trees
dreg= DecisionTreeRegressor(random_state=0)
dreg.fit(x_train,y_train)
y_pred1=dreg.predict(x_test)
print('Decision Tree')
r2.append(r2_score(y_test,y_pred1))
print('R2 Score',r2[0])
print('The root mean square error',np.sqrt(mean_squared_error(y_test,y_pred1)),'\n')
rmse.append(np.sqrt(mean_squared_error(y_test,y_pred1)))

##### Linear Regression
lreg=LinearRegression()
lreg.fit(x_train,y_train)
y_pred2=lreg.predict(x_test)
print('Linear Regression')
r2.append(r2_score(y_test,y_pred2))
print('R2 Score',r2[1])
print('The root mean square error',np.sqrt(mean_squared_error(y_test,y_pred2)),'\n')
rmse.append(np.sqrt(mean_squared_error(y_test,y_pred2)))

##### Random Forests
rreg=RandomForestRegressor(max_depth=15, random_state=0,n_estimators=1000)
rreg.fit(x_train,y_train)
y_pred3=rreg.predict(x_test)
print('Random Forests')
r2.append(r2_score(y_test,y_pred3))
print('R2 Score',r2[2])
print('The root mean square error',np.sqrt(mean_squared_error(y_test,y_pred3)),'\n')
rmse.append(np.sqrt(mean_squared_error(y_test,y_pred3)))

##### Artificial Neural Networks
nn=MLPRegressor(hidden_layer_sizes=(3,40),activation='relu',solver='adam',learning_rate='adaptive',max_iter=10000,learning_rate_init=0.01,alpha=0.01)
nn.fit(x_train,y_train)
y_pred4=nn.predict(x_test)
print('Artificial Neural Network')
r2.append(r2_score(y_test,y_pred4))
print('R2 Score',r2[3])
print('The root mean square error',np.sqrt(mean_squared_error(y_test,y_pred4)),'\n')
rmse.append(np.sqrt(mean_squared_error(y_test,y_pred4)))

### Ridge regression
rir= RidgeCV(alphas=[0.001,0.01,0.1,1,2,5,10,15,20,30], fit_intercept = False)
rir.fit(x_train,y_train)
y_pred5=rir.predict(x_test)
print('Ridge Regression')
r2.append(r2_score(y_test,y_pred5))
print('R2 Score',r2[4])
print('The root mean square error',np.sqrt(mean_squared_error(y_test,y_pred5)),'\n')
rmse.append(np.sqrt(mean_squared_error(y_test,y_pred5)))

##### LASSO Regression
lar = LassoCV(alphas=np.linspace(0,5,100))
lar.fit(x_train,y_train)
y_pred6=lar.predict(x_test)
print('Lasso Regression')
r2.append(r2_score(y_test,y_pred6))
print('R2 Score',r2[5])
print('The root mean square error',np.sqrt(mean_squared_error(y_test,y_pred6)),'\n')
rmse.append(np.sqrt(mean_squared_error(y_test,y_pred6)))

#####  Gradient boosting
gbr=ensemble.GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=3, max_features='sqrt',
                                       min_samples_leaf=15, min_samples_split=10, loss='huber')
gbr.fit(x_train,y_train)
y_pred7= gbr.predict(x_test)
print('Gradient Boosting')
r2.append(r2_score(y_test,y_pred7))
print('R2 Score',r2[6])
print('The root mean square error',np.sqrt(mean_squared_error(y_test,y_pred7)),'\n')
rmse.append(np.sqrt(mean_squared_error(y_test,y_pred7)))

# Final Plots
l=['DecT','RndF','LinR','ANN','RdgR','LSSR','GBR']
val=[0.20306,0.14574,0.13187,0.24829,0.12073,0.13149,0.12061]
l2=['DecT','LinR','RndF','ANN','RdgR','LSSR','GBR']
plt.bar(l2,rmse)
plt.xticks(rotation='vertical')
plt.xlabel('Model')
plt.ylabel('RMSE')
plt.title('Root Mean Squared Error from train test split')
plt.show()
plt.bar(l2,r2)
plt.xticks(rotation='vertical')
plt.xlabel('Model')
plt.ylabel('R^2')
plt.title('R^2 - Coefficient of Multiple Determination')
plt.show()
plt.bar(l,val)
plt.xticks(rotation='vertical')
plt.xlabel('Model')
plt.ylabel('RMSE')
plt.title('Root Mean Squared Error obtained from Kaggle')
plt.show()

import sys
import pandas as pd
import lib1738522 as lib

train = pd.read_csv(sys.argv[1], index_col=0)
test = pd.read_csv(sys.argv[2], index_col=0)
testr = test
#Dictionary of columns with outlier column names
outliers = {'LotArea':2,'MiscVal':6,'GrLivArea':4,'BsmtFinSF1':5,'BsmtFinSF2':3,'TotalBsmtSF':4,'stFlrSF':4,'ndFlrSF':4,'BsmtUnfSF':3}

#Creation of a list with categorical values to be converted to numericals
li =['MSZoning','Street', 'LotShape', 'LandContour',
     'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',
     'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
     'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond',
     'Foundation','BsmtQual','BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating',
     'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional',
     'GarageType','GarageFinish', 'GarageQual','GarageCond','PavedDrive','SaleType', 'SaleCondition']

def main():#main method to execute the overall code
    lib.data_predictor(train, test, li, outliers)


if __name__ == '__main__':
    main()












import pandas as pd
import numpy as np
from sklearn import linear_model,cross_validation
import matplotlib.pyplot as plt
from matplotlib.pyplot import style
from sklearn.preprocessing import LabelEncoder, StandardScaler, Imputer
style.use('ggplot')

imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0) # Instantiation of an imputer object to replace missing values
                                                            #with NAN with the mean of the column
def converter1(data, lis):
    n = LabelEncoder()#instatiation of LabelEcoder used to transform non-numerical labels (as long as they are hashable and comparable) to numerical
    data2 = data      #and unique numerical values to be accepted and used by the classifier
    for lis1 in lis:  #changes values for each column in data frame with name in the list
        data2[lis1] = n.fit_transform(data2[lis1].astype('str'))  # Fit label encoder and return encoded labels
    return data2

def outlier_remover(df, outliers):#remove outliers which deviate from values by a specific deviation
    for k in outliers:
        df = df[((df[k] - df[k].mean()) / df[k].std()).abs() < outliers[k]]#remove outliers which deviate from values by a specific deviation
    return df                                                              #this method employs mean and standard deviation


def data_preprocessor(df):
    # Drop columns having too many NaNs
    #df = converter1(df, li) #Convert categorical values to numbers
    df = df.select_dtypes([np.number]) # select only numbers from the data frame to be fit in the model
    df = df.fillna(df.median(axis=0), inplace=True) #replace column cells with no data with the median of the entire column
    df = df.rename(columns={'1stFlrSF': 'stFlrSF'}) #changing the column name of our dataframe
    df = df.rename(columns={'2ndFlrSF': 'ndFlrSF'}) #changing the column name of our dataframe
    return df


def data_predictor(train, test, li, outlier):
    l = test#to get an index from later
    train1 = train #just for plotting to show with outliers
    test = converter1(test, li)
    train = converter1(train,li)
    train = data_preprocessor(train)
    train = outlier_remover(train, outlier)
    test = data_preprocessor(test)

    clf = linear_model.Ridge(alpha=20.198273880198)#creation and instantiation of a classfier object
    X, y = train.values[:,:-1], train.values[:,-1]
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X) #'''creating a scalar object for transforming of our test data to conform to the training set data'''

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)#cross validation of data to avoid overfiting
    test = test.loc[:, train.columns[:-1]]  # select the variables that are in our model
    test = test.fillna(test.median(axis=0), inplace=True)  # fill NaNs
    test = scaler.transform(test)
    imp.fit(test)
    clf.fit(X_train, np.log(y_train))
    print(clf.score(X_train, np.log(y_train)))#checking the score of the clf on the training data set
    h = clf.score(X_test, np.log(y_test))#Score of the classifier on predicting 20% of the testing data
    preds = pd.DataFrame({"SalePrice": clf.predict(test)}, index=l.index)
    preds.SalePrice = np.exp(preds.SalePrice)
    preds.to_csv("pred.csv")  #results to submit
    print(h)# checking the score of the prediction of the test
    plots(train1, train, preds)


def plots(df, df2, df3): #function for ploting box plots of the data to check for outliers
    ax1 = plt.subplot(3, 1, 1)
    df.boxplot()
    plt.title('with outliers')
    plt.legend()
    ax2 = plt.subplot(3, 1, 2)
    df2.boxplot()
    plt.title('without outliers')
    plt.legend()
    ax3 = plt.subplot(3, 1, 3)
    df3.plot()
    plt.ylabel('Price Predictions')
    plt.legend()
    plt.show()


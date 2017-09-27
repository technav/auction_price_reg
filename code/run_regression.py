from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
import numpy as np

def rms_log_error(y_true, y_pred):
    ''' Calculate root mean squared log error between
        true values and predictions
    '''
    #import pdb; pdb.set_trace()
    log_diff = np.log(y_pred+1) - np.log(y_true+1)
    return np.sqrt(np.mean(log_diff**2))

def linearmodel(X, Y, train_index, fit_linear,a):
    lin_model = fit_linear(alpha=a, normalize=True).fit(X[train_index], Y[train_index])
    return lin_model

def add_machage(df):
    ''' Add a new series 'machine age' to the dataframe
    '''
    df['saleyear'] = pd.to_datetime(df['saledate']).apply(lambda d: d.year)
    df['machage'] = df['saleyear'] - df['YearMade']

def cross_validate(X, y, alphas, fit_linear, n_splits):
    X_scaled = preprocessing.scale(X)
    validation_error = []
    kf = KFold(n_splits=n_splits, shuffle=True)
    for i, a in enumerate(alphas):
        validation_error_fold = []
        for train_index, test_index in kf.split(X_scaled):
            print 'value of a: {}'.format(a)
            lin_model = linearmodel(X_scaled, y, train_index, fit_linear,a)
        #predict validation
            y_validation = lin_model.predict(X_scaled[test_index])
            validation_error_fold.append(rms_log_error(y[test_index], y_validation))
        validation_error.append(np.mean(validation_error_fold))
    min_key = validation_error.index(min(validation_error))
    return (alphas[min_key], validation_error[min_key])


if __name__ == '__main__':
    df = pd.read_csv('data/Train.csv')
    df = df.head(1000)
    df = df.set_index('SalesID').sort_index()
    y = np.array(df.SalePrice)
    #filter, type_change, replace_outliers, compute_age, nearest_avg, columns

    #Fill year 1000 with median for train/test
    med_tr = df[df['YearMade'] != 1000]['YearMade'].median()
    df['YearMade'].replace(1000, med_tr, inplace=True)

    #Compute machine age
    add_machage(df)

    #Filter columns based on eda and NAN values
    X = df[['machage', 'MachineHoursCurrentMeter', 'ProductGroup']]
    X = pd.concat([X[['machage', 'MachineHoursCurrentMeter']], \
                    pd.get_dummies(X['ProductGroup'])], axis=1)
    X = np.array(X)

    # Get train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    performance = {}
    alphas = [0.01, 0.1, 0.2, 0.5, 1]
    performance['ridge'] = cross_validate(X_train, y_train, alphas, Ridge, 10)

    # PART 2
    performance['lasso'] = cross_validate(X_train, y_train, alphas, Lasso, 10)

    #Using Pipeline and GridSearch instead
    p = Pipeline([('scaler', StandardScaler),('linmod', LinearRegression())]
    params  = {'alpha': [0.01, 0.05, 0.1, 1.0]}
    rms_le_scr = make_scorer(rms_log_error, greater_is_better=False)
    gs = GridSearchCV(p, params, scoring=rmsle_scr, cv = 10)
    best_model = gscv.fit(df.reset_index(), y, n_jobs=-1)

    # Get predictions
    df_test = pd.read_csv('data/test.csv')
    df_test = df_test.sort_values(by='SalesID')

    y_pred = best_model.predict(df_test)
    df_test['SalePrice'] = y_pred

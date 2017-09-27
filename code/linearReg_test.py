from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
import numpy as np

def mean_squared_log_error(y_true, y_pred):
    log_diff = np.log(y_pred+1) - np.log(y_true+1)
    return np.sqrt(np.mean(log_diff**2))

    #return mean_squared_error(np.log(y_true + 1), np.log(y_pred + 1))

# def linear_reg(X, Y):
#     # alphas = np.linspace(0.1,3)
#     # for i, a in enumerate(alphas):
#     #     X_data = preprocessing.scale(X)
#     #     fit = Lasso(alpha=a, normalize=True).fit(X, Y)
#     linear = LinearRegression()
#     linear.fit(X, Y)

def linearmodel(X, Y, train_index, fit_linear,a):
    lin_model = fit_linear(alpha=a, normalize=True).fit(X[train_index], Y[train_index])
    return lin_model

def cross_validate(df, columns, alphas, fit_linear,n_splits):
    X = df[columns]
    y = df['SalePrice']
    X_scaled = preprocessing.scale(X)
    validation_error = []
    kf = KFold(n_splits=n_splits, shuffle=True)
    for i, a in enumerate(alphas):
        validation_error_fold = []
        for train_index, test_index in kf.split(X_scaled):
            lin_model = linearmodel(X_scaled, y, train_index, fit_linear,a)
        #predict validation
            y_validation = lin_model.predict(X_scaled[test_index])
            validation_error_fold.append(mean_squared_log_error(y[test_index], y_validation))
        validation_error.append(np.mean(validation_error_fold))
    # plt.scatter(alphas, train_error)
    # plt.scatter(alphas, validation_error)
    # plt.show()
    min_key = validation_error.index(min(validation_error))
    return (alphas[min_key], validation_error[min_key])



# def get_values(test,model):
#     pd.DataFrame(test['SalesID'],)
#
# if __name__ == '__main__':
#     median_machine_hours_current_meter = df[df['MachineHoursCurrentMeter'] > 0]['MachineHoursCurrentMeter'].median()
#
# def set_0_as_median(machine_hours, median):
#
#
#     alphas = np.logspace(-2, 2)
#     performance['ridge'] = cross_validate(X_train, y_train, Ridge, alphas)
#
#     # PART 2
#     alphas = np.linspace(0.1,3)
#     performance['lasso'] = cross_validate(X_train, y_train, Lasso, alphas)
#
#
#     # PART 3
#     performance['olr'] = cross_validate(X_train, y_train, Lasso, [0])
#
#     print performance

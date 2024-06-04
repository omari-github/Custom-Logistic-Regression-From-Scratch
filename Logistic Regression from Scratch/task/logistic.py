import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from scipy.sparse.base import spmatrix
from sklearn.linear_model import LogisticRegression
from decimal import Decimal



class CustomLogisticRegression:
    coef_ = np.array([])
    bias_ = 0
    mse_error_first = None
    mse_error_last = None
    logloss_error_first = None
    logloss_error_last = None

    def standardize(self, X):
        # std: object = np.std(X, axis=1)
        # mean = np.mean(X, axis=1)
        # mean = mean.reshape(-1, 1)
        mean = X.mean(0)
        std = X.std(0)
        # mean = mean[:, np.newaxis]
        # std = std[:, np.newaxis]
        X: object = (X - mean) / std
        return X

    def __init__(self, fit_intercept=True, l_rate=0.01, n_epoch=100):
        self.fit_intercept = fit_intercept
        self.l_rate = l_rate
        self.n_epoch = n_epoch

    def sigmoid(self, t: float) -> float:
        return 1 / (1 + np.exp(-t))

    def predict_proba(self, row, coef_):
        #if isinstance(row, np.ndarray) and isinstance(coef_, np.ndarray):
        #std = np.std(row)
        #mean = np.mean(row)
        #row = (row - mean) / std
        #print('row {} mean {} std {}'.format(row, mean, std))
        #t = np.dot(row, coef_) + self.bias_
        t = np.dot(row, coef_) + self.bias_
        return self.sigmoid(t)

    def fit_mse(self, X_train, y_train):
        X_train_arr = X_train.to_numpy()
        y_train_arr = y_train.to_numpy()
        #self.coef_ = np.random.random(X_train_arr.shape[1])  # initialized weights
        self.coef_ = np.zeros(X_train_arr.shape[1])
        y_hat = np.array(y_train)
        N = X_train_arr.shape[0]

        for _ in range(self.n_epoch):
            for i, row in enumerate(X_train.to_numpy()):
                y_hat[i] = self.predict_proba(row, self.coef_)
                # update all weights
                self.bias_ = self.bias_ - (self.l_rate * (y_hat[i] - y_train_arr[i]) * y_hat[i] * (1 - y_hat[i]))
                # self.bias_ = self.bias_ - (self.l_rate * 2 * (y_hat[i] - y_train_arr[i]))
                for j in range(len(self.coef_)):
                    self.coef_[j] = self.coef_[j] - (
                                self.l_rate * (y_hat[i] - y_train_arr[i]) * y_hat[i] * (1 - y_hat[i]) * X_train_arr[i][
                            j])
                # err: object = np.sum(np.square(np.diff(y_hat, y_train_arr, axis=0))) / N
                # if (_ == 0 | _ == self.n_epoch - 1):
            if (_ == 0):
                self.mse_error_first = (np.square(y_hat - y_train_arr)) / N
                    # print('err:{}'.format(err.tolist()))
            if (_ == self.n_epoch - 1):
                self.mse_error_last = (np.square(y_hat - y_train_arr)) / N

    def fit_log_loss(self, X_train, y_train):
        X_train_arr = X_train.to_numpy()
        y_train_arr = y_train.to_numpy()
        # self.coef_ = np.random.random(X_train_arr.shape[1])  # initialized weights
        self.coef_ = np.zeros(X_train_arr.shape[1])
        N = X_train_arr.shape[0]
        y_hat = np.array(y_train)

        for _ in range(self.n_epoch):
            for i, row in enumerate(X_train.to_numpy()):
                y_hat[i] = self.predict_proba(row, self.coef_)
                # update all weights
                self.bias_ = self.bias_ - (self.l_rate * (y_hat[i] - y_train_arr[i]) / N)
                # self.bias_ = self.bias_ - (self.l_rate * 2 * (y_hat[i] - y_train_arr[i]))
                for j in range(len(self.coef_)):
                    self.coef_[j] = self.coef_[j] - (
                                        self.l_rate * (y_hat[i] - y_train_arr[i]) * X_train_arr[i][j]) / N
                    # if (_ == 0 | _ == self.n_epoch - 1):
            if (_ == 0):
                self.logloss_error_first = (np.log(y_hat) * y_train_arr) + ((1 - y_train_arr) * np.log(1 - y_hat)) * -1 / N
            if (_ == self.n_epoch - 1):
                self.logloss_error_last = (np.log(y_hat) * y_train_arr) + ((1 - y_train_arr) * np.log(1 - y_hat)) * -1 / N

                    # err = np.sum(np.dot(np.log(y_hat), y_train_arr), np.dot(np.diff(1, y_train_arr), np.log(np.diff(1, y_hat))))

    def predict(self, X_test, cut_off=0.5):
        X_test_arr = X_test.to_numpy()
        return np.array([round(self.predict_proba(X_test_arr[i], self.coef_)) for i in range(len(X_test_arr))])
        # return predictions # predictions are binary values - 0 or 1

cancer = load_breast_cancer()
clr = CustomLogisticRegression(fit_intercept=True, l_rate=0.01, n_epoch=1000)
df = pd.DataFrame(data=np.c_[clr.standardize(cancer.data), cancer.target],
columns=np.append(cancer.feature_names, ['target']))

X_train, X_test, y_train, y_test = train_test_split(
df[['worst concave points', 'worst perimeter', 'worst radius']], df['target'], test_size=0.2,random_state=43)

lr = LogisticRegression(fit_intercept=True, l1_ratio=0.01)
lr.fit(X_train, y_train)
predictions = lr.predict(X_test)
sklearn_accuracy = (len(predictions) - np.sum(np.logical_xor(np.round(predictions), y_test.to_numpy()))) / len(predictions)


# print([lr.predict_proba(np.array([1, X_test_arr[i][0], X_test_arr[i][1]]), coef_) for i in range(10)])
X_test_arr = X_test.to_numpy()

clr.fit_mse(X_train, y_train)
predictions = clr.predict(X_test)
# mse_err = (np.square(predictions  - y_test.to_numpy())) / len(predictions)
# print('err:{}'.format(mse_err.tolist()))
mse_accuracy = (len(predictions) - np.sum(
np.logical_xor(np.round(predictions), y_test.to_numpy()))) / len(predictions)
# print({'mse_accuracy': mse_accuracy, 'mse_error_first': clr.mse_error_first, 'mse_error_last': clr.mse_error_last})

clr.fit_log_loss(X_train, y_train)
predictions = clr.predict(X_test)
# logloss_err =  (np.log(predictions) * y_test.to_numpy()) + ((1 -  y_test.to_numpy()) * np.log(1 - predictions)) * -1/N
# print('logloss_err:{}\n\n'.format(logloss_err.tolist()))
logloss_accuracy = (len(predictions) - np.sum(
np.logical_xor(np.round(predictions), y_test.to_numpy()))) / len(predictions)
# print({'logloss_accuracy': logloss_accuracy, 'logloss_error_first': clr.logloss_error_first, 'logloss_error_last': clr.logloss_error_last})

# print({'coef_': [lr.bias_] + lr.coef_.tolist(), 'accuracy': (len(predictions) - np.sum(np.logical_xor(np.round(predictions), y_test.to_numpy())))/len(predictions)})
# print('epoch: {}'.format(lr.n_epoch))
# print({'coef_': [round(clr.bias_, 8)] + np.round(clr.coef_, 8).tolist(), 'accuracy': round((len(predictions) - np.sum(np.logical_xor(np.round(predictions), y_test.to_numpy()))) / len(predictions), 2)})

print({'mse_accuracy': mse_accuracy, 'logloss_accuracy': logloss_accuracy,\
       'sklearn_accuracy': sklearn_accuracy, 'mse_error_first': clr.mse_error_first.tolist(),\
       'mse_error_last': clr.mse_error_last.tolist(),\
       'logloss_error_first': clr.logloss_error_first.tolist(),\
       'logloss_error_last': clr.logloss_error_last.tolist()})

#print('Answers to the questions:\n "1)" {}\n "2)" {}\n "3)" {}\n  "4)" {}\n "5)" {}\n "6)" {}'.\
print('Answers to the questions:\n 1) {}\n 2) {}\n 3) {}\n 4) {}\n 5) {}\n 6) {}'.\
format(round(Decimal(round(clr.mse_error_first.min(), 5)), 5),\
       round(clr.mse_error_last.min(), 5),\
       #0.00153,\
       round(clr.logloss_error_first.max(), 5),\
       #round(Decimal(round(clr.logloss_error_first.max(), 5)), 5),\
       round(clr.logloss_error_last.max(), 3),\
       #0.00600,\
       'expanded' if (clr.mse_error_last.max() - clr.mse_error_last.min()) > (clr.mse_error_first.max() - clr.mse_error_first.min()) else 'narrowed',\
       'expanded' if (clr.logloss_error_last.max() - clr.logloss_error_last.min()) > (clr.logloss_error_first.max() - clr.logloss_error_first.min()) else 'narrowed'\
      )\
     )

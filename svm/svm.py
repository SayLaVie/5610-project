"""
This code follows the examples at
https://www.kaggle.com/nishan192/mnist-digit-recognition-using-svm
"""

# import numpy as np
import pandas as pd
# from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
# from sklearn import metrics
# from sklearn.model_selection import validation_curve
# from sklearn.model_selection import KFold
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import scale
# import matplotlib.pyplot as plt
# import seaborn as sns

train_data = pd.read_csv("../data/train.csv")
test_data = pd.read_csv("../data/test.csv")

# sns.countplot(train_data["label"])
# plt.show()

# Plotting a sample
# four = train_data.iloc[3, 1:]
# four = four.values.reshape(28, 28)
# plt.imshow(four, cmap='gray')
# plt.show()

# Normalization
y = train_data['label']
X = train_data.drop(columns='label')
X = X / 255.0
test_data = test_data / 255.0
X_scaled = scale(X)

# train test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X_scaled, y, test_size=0.3, train_size=0.2, random_state=10
#     )

# linear model
model_linear = SVC(kernel='linear')
# model_linear.fit(X_train, y_train)
model_linear.fit(X, y)
# y_pred = model_linear.predict(X_test)
y_pred = model_linear.predict(test_data)

print(type(y_pred))
print(len(y_pred))
print(y_pred.shape)

# print as csv
y_pred_df = pd.DataFrame(y_pred, columns=["Label"])
y_pred_df["ImageId"] = range(1, 28001)
y_pred_df.to_csv(r'linear_pred.csv')

# confusion matrix and accuracy
# linear_accuracy = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
# linear_confusion_matrix = metrics.confusion_matrix(y_true=y_test, y_pred=y_pred)

# print(linear_accuracy)
# print(linear_confusion_matrix)

# non-linear model
# non_linear_model = SVC(kernel='rbf')
# non_linear_model.fit(X_train, y_train)
# y_pred = non_linear_model.predict(X_test)

# non_linear_accuracy = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
# non_linear_confusion_matrix = metrics.confusion_matrix(
#     y_true=y_test, y_pred=y_pred
#     )

# print(non_linear_accuracy)
# print(non_linear_confusion_matrix)
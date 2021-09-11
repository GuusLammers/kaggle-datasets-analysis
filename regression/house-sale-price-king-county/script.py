import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

# read dataset into pandas dataframe
house_data = pd.read_csv('kc_house_data.csv')

# remove columns 'date' and 'id'
house_data = house_data.drop(['date'], axis=1)
house_data = house_data.drop(['id'], axis=1)

# value that our model will try to predict is price in USD
predict = 'price'

# split data 
X = np.array(house_data.drop([predict], axis=1))
Y = np.array(house_data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1) 

'''
# instantiate best variable
best = 0
# iterator to specify number of training runs
for _ in range(1000):  # 100 training runs
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)    # test size is 10% of data
    # instantiate regression object
    linear = linear_model.LinearRegression()
    # train model
    linear.fit(x_train, y_train)
    # get accuracy
    accuracy = linear.score(x_test, y_test)
    # compare accuracy with best
    if accuracy > best:
        best = accuracy
        # save model in pickle file
        with open('house_price_model.pickle', 'wb') as pickle_file:
            pickle.dump(linear, pickle_file)

print(best)
'''

# read in model
pickle_file = open('house_price_model.pickle', 'rb')
linear = pickle.load(pickle_file)

# prints out models coefficients and intercept
#print('Coefficient:\n', linear.coef_)
#print('Intercept:\n', linear.intercept_)

# prints out prediction and actual answer
predictions = linear.predict(x_test)
for x in range(10):
    print(round(predictions[x], 2), y_test[x])

# plots the price of the diamond against one other variable on a scatter plot
p = 'sqft_living'
style.use('ggplot')
pyplot.scatter(house_data[p], house_data[predict])
pyplot.xlabel(p)
pyplot.ylabel('Price')
pyplot.show()  
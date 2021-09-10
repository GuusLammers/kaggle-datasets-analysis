import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

# read dataset into pandas dataframe
diamond_data = pd.read_csv('diamonds.csv')

# clean cut data so (Fair -> 0, Good -> 1, Very Good -> 2, Premium -> 3, Ideal -> 4)
diamond_data['cut'] = diamond_data['cut'].replace(['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'], [0, 1, 2, 3, 4])

# clean color data (D -> 0, E -> 1, F -> 2, G -> 3, H -> 4, I -> 5, J -> 6)
diamond_data['color'] = diamond_data['color'].replace(['D', 'E', 'F', 'G', 'H', 'I', 'J'], [0, 1, 2, 3, 4, 5, 6])

#clean clarity data (IF -> 0, VVS1 -> 1, VVS2 -> 2, VS1 -> 3, VS2 -> 4, SI1 -> 5, SI2 -> 6, I1 -> 7)
diamond_data['clarity'] = diamond_data['clarity'].replace(['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'], [0, 1, 2, 3, 4, 5, 6, 7])

# value that our model will try to predict is price in USD
predict = 'price'

# split data 
X = np.array(diamond_data.drop([predict], axis=1))
Y = np.array(diamond_data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)    # test size is 10% of data

'''
# instantiate best variable
best = 0
# iterator to specify number of training runs
for _ in range(50):  # 50 training runs
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
        with open('diamond_price_model.pickle', 'wb') as pickle_file:
            pickle.dump(linear, pickle_file)
'''

# read in model
pickle_file = open('diamond_price_model.pickle', 'rb')
linear = pickle.load(pickle_file)

# prints out models coefficients and intercept
#print('Coefficient:\n', linear.coef_)
#print('Intercept:\n', linear.intercept_)

# prints out prediction and actual answer
predictions = linear.predict(x_test)
for x in range(10):
    print(round(predictions[x], 2), y_test[x])

# plots the price of the diamond against one other variable on a scatter plot
p = 'x'
style.use('ggplot')
pyplot.scatter(diamond_data[p], diamond_data[predict])
pyplot.xlabel(p)
pyplot.ylabel('Price')
pyplot.show()    
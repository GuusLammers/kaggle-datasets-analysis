import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing, linear_model
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

# read dataset into pandas dataframe
mushroom_data = pd.read_csv('mushrooms.csv')

# instantiate preprocessing label encoder object
label_encoder = preprocessing.LabelEncoder()

# encode all columns containing text
clas = label_encoder.fit_transform(list(mushroom_data['class']))
cap_shape =  label_encoder.fit_transform(list(mushroom_data['cap-shape']))
cap_surface = label_encoder.fit_transform(list(mushroom_data['cap-surface']))
cap_color = label_encoder.fit_transform(list(mushroom_data['cap-color']))
bruises = label_encoder.fit_transform(list(mushroom_data['bruises']))
odor = label_encoder.fit_transform(list(mushroom_data['odor']))
gill_attachment = label_encoder.fit_transform(list(mushroom_data['gill-attachment']))
gill_spacing = label_encoder.fit_transform(list(mushroom_data['gill-spacing']))
gill_size = label_encoder.fit_transform(list(mushroom_data['gill-size']))
gill_color = label_encoder.fit_transform(list(mushroom_data['gill-color']))

# variable that is being predicted
predicted = 'class'

# features and labels
X = list(zip(cap_shape, cap_surface, cap_color, bruises, odor, gill_attachment, gill_spacing, gill_size, gill_color))
Y = list(clas)

# splite data into train and test
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1) 

'''
# intantiate tracking variables
best_accuracy = 0
best_number_neighbors = 1
# iterate through odd numbers between 1-30 to test for the ideal number of neighbors
for i in range(1, 30, 2):
	# iterate 50 times
	for _ in range(50):
		# instantiate knn object
		knn = KNeighborsClassifier(n_neighbors=i)
		# train model
		knn.fit(x_train, y_train)
		# get accuracy
		accuracy = knn.score(x_test, y_test)
		# check if accuracy is greater than best_accuracy
		if accuracy > best_accuracy:
			# update best_accuracy and nest_number_neighbors
			best_accuracy = accuracy
			best_number_neighbors = i
			# save model
			with open('mushroom_model.pickle', 'wb') as pickle_file:
				pickle.dump(knn, pickle_file)
'''

# read in model
pickle_file = open('mushroom_model.pickle', 'rb')
linear = pickle.load(pickle_file)

# prints out prediction and actual answer
names = ['Edible', 'Poisonous']
predictions = linear.predict(x_test)
for x in range(100):
    print('Prediction: ', names[predictions[x]], 'Answer: ', names[y_test[x]])


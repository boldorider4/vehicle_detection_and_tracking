import time
import pickle
import os
import sys
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from load_set_extract_feat import *
from parameters import *


def train_classifier():
  # if not already done, load training images and extract features
  if not os.path.isfile('features.' + color_space.lower() + '.p'):
    load_and_extract()

  # Restore previously saved classifier and scaler
  with open('features.' + color_space.lower() + '.p', 'rb') as pf:
    pickle_features = pickle.load(pf)
    car_features = pickle_features["car_features"]
    notcar_features = pickle_features["notcar_features"]
    udacity_cars_features = pickle_features["udacity_cars_features"]
    print('features restored successfully!')
    pf.close()

  #udacity_cars_features = udacity_cars_features[0:int(len(udacity_cars_features)/3)]

  # flags for enabling training with various classifiers
  fancy_training = False
  grid_search_training = False

  #X = np.vstack((car_features, notcar_features, udacity_cars_features)).astype(np.float64)
  X = np.vstack((car_features, notcar_features)).astype(np.float64)

  # Fit a per-column scaler
  X_scaler = StandardScaler().fit(X)
  # Apply the scaler to X
  scaled_X = X_scaler.transform(X)

  # Define the labels vector
  #y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features)), np.ones(len(udacity_cars_features))))
  y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

  # Split up data into randomized training and test sets
  rand_state = np.random.randint(0, 100)
  X_train, X_test, y_train, y_test = train_test_split(
      scaled_X, y, test_size=0.2, random_state=rand_state)

  print('Training classifier... (this could take a while depending on chosen classifier)')
  print('Feature vector length:', len(X_train[0]))

  if fancy_training:
    # enable grid search
    if grid_search_training:
      svr = SVC()
      # try radial basis function, polynomial, various C and gamma values
      search_parameters = { 'kernel': ('rbf', 'poly'),
                            'C' : np.logspace(-1,1,3),
                            'gamma': np.logspace(-7,1,3) }
      clf = GridSearchCV(svr, search_parameters, scoring='accuracy', verbose=10)
    else:
      # Use a radial basis function
      clf = SVC(kernel='rbf')
  else:
    # Use a linear SVC
    clf = LinearSVC()


  # Check the training time
  t=time.time()
  clf.fit(X_train, y_train)
  t2 = time.time()
  print(round(t2-t, 2), 'Seconds to train classifer...')

  # Check the score of the classifier
  print('Test Accuracy of classifier = ', round(clf.score(X_test, y_test), 4))

  # Saving classifier to pickle file
  pickle_clf = { "clf" : clf, "X_scaler" : X_scaler }
  with open('clf_scaler.' + color_space.lower() + '.p', 'wb') as pf:
    pickle.dump(pickle_clf, pf)
    print('classifier saved successfully!')
    pf.close()

if __name__ == '__main__':
  train_classifer()
  sys.exit()

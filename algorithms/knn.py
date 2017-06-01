import numpy as np
import scipy.io as scio
from sklearn import neighbors

offline_data = scio.loadmat('offline_data_random.mat')
online_data = scio.loadmat('online_data.mat')
offline_location, offline_rss = offline_data['offline_location'], offline_data['offline_rss']
trace, rss = online_data['trace'][0:1000, :], online_data['rss'][0:1000, :]
del offline_data
del online_data

def accuracy(predictions, labels):
    return np.mean(np.sqrt(np.sum((predictions - labels)**2, 1)))


knn_reg = neighbors.KNeighborsRegressor(40, weights='uniform', metric='euclidean')
knn_reg.fit(offline_rss, offline_location)
predictions = knn_reg.predict(rss)
acc = accuracy(predictions, trace)
print ("accuracy: ", acc/100, "m")

from sklearn import ensemble
from sklearn.multioutput import MultiOutputRegressor
clf = MultiOutputRegressor(ensemble.GradientBoostingRegressor(n_estimators=100, max_depth=10))
clf.fit(offline_rss, offline_location)
predictions = clf.predict(rss)
acc = accuracy(predictions, trace)
print ("accuracy: ", acc/100, "m")
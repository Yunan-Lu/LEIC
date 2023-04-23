import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.io import loadmat
from leic import LEIC
from utils import report, binarize

# load dataset
dataset = '3dfe'
data = loadmat('datasets/%s.mat' % dataset)
X, D = data['features'], data['label_distribution']
X = MinMaxScaler().fit_transform(X)
L = binarize(D)

# load the optimal hyperparameters
p = np.load('config.npy', allow_pickle=True).item()[dataset]
p['verbose'] = 10

# train LEIC
leic = LEIC(**p).fit(X, L)
Drec = leic.label_distribution_

# report the results
report(Drec, D, ds=dataset)
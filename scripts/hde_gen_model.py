import os 
import sys
import glob
import pickle
import argparse
import numpy as np 
import pyemma as py
import tensorflow as tf
from os.path import join
import keras.backend as K
from sklearn.model_selection import KFold, train_test_split
import sklearn.preprocessing as pre
import keras.callbacks as callbacks

import sys
sys.path.append('/home/hsidky/Code/hde/')
from hde import HDE
from hde import analysis


parser = argparse.ArgumentParser()
parser.add_argument('--prefix', type=str)
parser.add_argument('--n_models', type=int)
parser.add_argument('--n_chunks', type=int)
parser.add_argument('--n_components', type=int)
parser.add_argument('--lag_time', type=int)
parser.add_argument('--hde_lag', type=int)
parser.add_argument('--score_k', type=int)
parser.add_argument('--max_iter', type=int)
parser.add_argument('--device_id', type=str)
args = parser.parse_args()


prefix = args.prefix
n_models = args.n_models
n_components = args.n_components
lag_time = args.lag_time
score_k = args.score_k
hde_lag = args.hde_lag
max_iter = args.max_iter
device_id = args.device_id

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = device_id

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
K.tensorflow_backend.set_session(tf.Session(config=config))

# Create chunked data.
all_data = np.load('all_data.npy')
n_dim = all_data.shape[1]
n_chunks = args.n_chunks
n_data = all_data.shape[0]
n_batch = n_data//n_chunks
chunked_data = [all_data[i:i + n_batch] for i in range(0, n_data, n_batch)]


k_folds = KFold(n_splits=n_models, shuffle=True)

#calls = [callbacks.EarlyStopping(patience=30, restore_best_weights=True)]

model_files = []
#model_train_scores = []
#model_test_scores = []
model_score = []

for i in range(n_models): 
    print('Processing model {}'.format(i))

    #train_data, test_data = train_test_split(chunked_data, test_size=0.3)

    hde = HDE(
        n_dim, 
        n_components=n_components, 
        n_epochs=max_iter, 
        lag_time=hde_lag, 
        batch_size=500000,
        #callbacks=calls,
        learning_rate=0.01, 
        batch_normalization=True,
        verbose=False
    )
    
    
    try:
        #hde.fit(train_data, test_data)
        hde.fit(all_data)	
        score = hde.score(all_data)
        #z_train = [hde.transform(x) for x in train_data]
        #z_test = [hde.transform(x) for x in test_data]
        
        #train_score = hde.score(train_data, lag_time=lag_time, score_k=score_k)
        #test_score = hde.score(test_data, lag_time=lag_time, score_k=score_k) 

    except:
        print('Oops. Something went wrong on fold {}.'.format(i))

    #model_train_scores.append(train_score)
    #model_test_scores.append(test_score)
    model_score.append(score)

    hde.callbacks = None
    hde.history = None
 
    filename = '{}_model_{:03d}.pkl'.format(prefix, i)
    pickle.dump(hde, open(filename, 'wb'))
    model_files.append(filename)


    #K.clear_session()

    pickle.dump([model_score, model_files], open('model_data.pkl', 'wb'))

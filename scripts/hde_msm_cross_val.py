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
parser.add_argument('--n_folds', type=int)
parser.add_argument('--n_chunks', type=int)
parser.add_argument('--n_components', type=int)
parser.add_argument('--n_cluster', type=int)
parser.add_argument('--lag_time', type=int)
parser.add_argument('--hde_lag', type=int)
parser.add_argument('--score_k', type=int)
parser.add_argument('--max_iter', type=int)
args = parser.parse_args()


prefix = args.prefix
n_folds = args.n_folds
n_components = args.n_components
n_cluster = args.n_cluster
lag_time = args.lag_time
score_k = args.score_k
hde_lag = args.hde_lag
max_iter = args.max_iter

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="1" 

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3
K.tensorflow_backend.set_session(tf.Session(config=config))

#trj_dir = "/home/hsidky/Data/DESRES-Trajectory_2JOF-0-protein/2JOF-0-protein"
#pdb_file = os.path.join(trj_dir, "2JOF-0-protein.pdb")
#traj_files = glob.glob(join(trj_dir, '2JOF-0-protein-*.dcd'))
#traj_files.sort()

#combined_feat = py.coordinates.featurizer(pdb_file)
#combined_feat.add_backbone_torsions(cossin=True, periodic=False)
#combined_feat.add_sidechain_torsions(cossin=True, which=["chi1"], periodic=False)
#combined_feat.add_distances_ca(periodic=False)
#combined_data = py.coordinates.load(traj_files, features=combined_feat)

#n_dim = combined_feat.dimension()
#scale = pre.MinMaxScaler(feature_range=(-1, 1))
#scale.fit(np.concatenate(combined_data))

# Create chunked data.
#all_data = scale.transform(np.concatenate(combined_data))
all_data = np.load('all_data.npy')
n_dim = all_data.shape[1]
n_chunks = args.n_chunks
n_data = all_data.shape[0]
n_batch = n_data//n_chunks
chunked_data = [all_data[i:i + n_batch] for i in range(0, n_data, n_batch)]


k_folds = KFold(n_splits=n_folds, shuffle=True)
calls = [callbacks.EarlyStopping(patience=30, restore_best_weights=True)]

train_scores = np.zeros(n_folds)
test_scores = np.zeros(n_folds)
hde_timescales = []
msm_timescales = []

for i in range(n_folds):
    #train_val_data = [chunked_data[j] for j in train_idx]
    #test_data = [chunked_data[j] for j in test_idx]
    
    print('Processing fold {}'.format(i))

    train_data, test_data = train_test_split(chunked_data, test_size=0.5)

    hde = HDE(
        n_dim, 
        n_components=n_components, 
        n_epochs=max_iter, 
        lag_time=hde_lag, 
        batch_size=500000,
        #callbacks=calls,
        learning_rate=0.01, 
        batch_normalization=True,
        latent_space_noise=0.0,
        verbose=False
    )
        
    hde.fit(train_data)
    hde_timescales.append(hde.timescales_)
        
    z_train = [hde.transform(x) for x in train_data]
    z_test = [hde.transform(x) for x in test_data]

    cluster = py.coordinates.cluster_kmeans(z_train, k=n_cluster, max_iter=50, stride=1, n_jobs=1)
    train_dtrajs = cluster.dtrajs
    test_dtrajs = cluster.assign(z_test)
    
    msm = py.msm.bayesian_markov_model(cluster.dtrajs, lag=lag_time)
    msm_timescales.append(msm.sample_mean('timescales', k=50))
    
    train_scores[i] = msm.score(train_dtrajs, score_k=score_k, score_method='VAMP2')
    test_scores[i] = msm.score(test_dtrajs, score_k=score_k, score_method='VAMP2')
    
    K.clear_session()

    #pickle.dump([train_scores, test_scores], open('{}_msm_scores_lag_{}_k_{}_components_{}_cluster_{}_hlag_{}_iter_{}.pkl'.format(prefix, lag_time, n_folds, n_components, n_cluster, hde_lag, max_iter), 'wb'))
    pickle.dump([train_scores, test_scores], open('{}_msm_lag_{}_k_{}_components_{}_cluster_{}_hlag_{}.pkl'.format(prefix, lag_time, n_folds, n_components, n_cluster, hde_lag), 'wb'))


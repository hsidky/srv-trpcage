import os 
import sys
import glob
import pickle
import argparse
import numpy as np 
import pyemma as py
from os.path import join
from sklearn.model_selection import KFold, train_test_split
import sklearn.preprocessing as pre


parser = argparse.ArgumentParser()
parser.add_argument('--prefix', type=str)
parser.add_argument('--n_folds', type=int)
parser.add_argument('--n_components', type=int)
parser.add_argument('--n_chunks', type=int)
parser.add_argument('--tica_lag', type=int)
parser.add_argument('--msm_lag', type=int)
parser.add_argument('--score_k', type=int)
parser.add_argument('--full_basis', type=bool, default=True)
args = parser.parse_args()


prefix = args.prefix
n_folds = args.n_folds
n_components = args.n_components
tica_lag = args.tica_lag
msm_lag = args.msm_lag
score_k = args.score_k
full_basis = args.full_basis
n_cluster = 200


trj_dir = "/home/hsidky/Data/DESRES-Trajectory_2JOF-0-protein/2JOF-0-protein"
pdb_file = os.path.join(trj_dir, "2JOF-0-protein.pdb")
traj_files = glob.glob(join(trj_dir, '2JOF-0-protein-*.dcd'))
traj_files.sort()

combined_feat = py.coordinates.featurizer(pdb_file)

if full_basis:
    combined_feat.add_backbone_torsions(cossin=True, periodic=False)
    combined_feat.add_sidechain_torsions(cossin=True, which=["chi1"], periodic=False)
combined_feat.add_distances_ca(periodic=False)
combined_data = py.coordinates.load(traj_files, features=combined_feat)

n_dim = combined_feat.dimension()

# Create chunked data.
scale = pre.MinMaxScaler(feature_range=(-1, 1))
scale.fit(np.concatenate(combined_data))

all_data = scale.transform(np.concatenate(combined_data))
n_chunks = args.n_chunks
n_data = all_data.shape[0]
n_batch = n_data//n_chunks
chunked_data = [all_data[i:i + n_batch] for i in range(0, n_data, n_batch)]


k_folds = KFold(n_splits=n_folds, shuffle=True)

train_scores = np.zeros(n_folds)
test_scores = np.zeros(n_folds)
tica_timescales = []
msm_timescales = []

for i in range(n_folds):
    train_data, test_data = train_test_split(chunked_data, test_size=0.5)

    tica = py.coordinates.tica(data=train_data, lag=tica_lag, dim=n_components, kinetic_map=True)
    tica_timescales.append(tica.timescales)
    
    z_train = tica.get_output()
    z_test = tica.transform(test_data)
    
    cluster = py.coordinates.cluster_kmeans(z_train, k=n_cluster, max_iter=50, stride=1, n_jobs=1)
    train_dtrajs = cluster.dtrajs
    test_dtrajs = cluster.assign(z_test)
    
    msm = py.msm.bayesian_markov_model(cluster.dtrajs, lag=msm_lag)
    msm_timescales.append(msm.sample_mean('timescales', k=15))
    
    train_scores[i] = msm.score(train_dtrajs, score_k=score_k, score_method='VAMP2')
    test_scores[i] = msm.score(test_dtrajs, score_k=score_k, score_method='VAMP2')

    pickle.dump([train_scores, test_scores], open('{}_msm_scores_lag_{}_k_{}_components_{}_tlag_{}.pkl'.format(prefix, msm_lag, n_folds, n_components, tica_lag), 'wb'))
    pickle.dump([msm_timescales, tica_timescales], open('{}_msm_timescales_lag_{}_k_{}_components_{}_tlag_{}.pkl'.format(prefix, msm_lag, n_folds, n_components, tica_lag), 'wb'))


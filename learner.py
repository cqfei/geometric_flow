import numpy as np
import torch
import traceback
from itertools import product
from evaluation import evaluate
from load_data import Data
from model import RicciFlowTrainer

from sklearn.manifold import (
    LocallyLinearEmbedding,
    SpectralEmbedding,
)

from sklearn.manifold._spectral_embedding import spectral_embedding
from scipy.sparse import csr_matrix

from log import setup_logger,change_log_file
logger = setup_logger('my_logger', f'./logs/log.log')
class Trainer:
    def __init__(self, data_home='data/manifold_learning/'):
        self.data_home = data_home


    def train_baselines(self,dataset):
        # get the baseline results of the dataset
        args = {
                'USPS':{'n_neighbors_list': [45],"n_components":8},
            'autoUniv_au6_1000':{'n_neighbors_list': [45],"n_components":8},
                'gender':{'n_neighbors_list': [10],"n_components":2},
                'student_prediction':{ 'n_neighbors_list': [45],"n_components":8},
                'User_Knowledge':{'n_neighbors_list': [12],"n_components":2},
            'yeast':{'n_neighbors_list': [10], "n_components": 2},
            'minist':{'n_neighbors_list': [45],"n_components":8},
        }
        arg = args[dataset]
        log = f'log/baselines.log'
        change_log_file(logger,log)
        X, y = Data(self.data_home).load_data(dataset)
        logger.info(f'dataset: {dataset}, n_neighbors_list: {str(arg["n_neighbors_list"])}')
        for n_neighbors in arg["n_neighbors_list"]:
                logger.info(f'n_neighbors: {n_neighbors}')
                embeddings = {
                    "Spectral embedding": SpectralEmbedding(
                        n_components=arg["n_components"], random_state=0, eigen_solver="arpack",
                        n_neighbors=n_neighbors
                    ),
                    "Standard LLE embedding": LocallyLinearEmbedding(
                        n_neighbors=n_neighbors, n_components=arg["n_components"], method="standard", eigen_solver='dense'
                    ),
                    "Hessian LLE embedding": LocallyLinearEmbedding(
                        n_neighbors=n_neighbors, n_components=arg["n_components"], method="hessian", eigen_solver='dense'
                    ),
                    "LTSA LLE embedding": LocallyLinearEmbedding(
                        n_neighbors=n_neighbors, n_components=arg["n_components"], method="ltsa", eigen_solver='dense'
                    ),

                }
                projections = {}
                for name, transformer in embeddings.items():
                    logger.info(f'algorithm:{name}')
                    projections[name] = transformer.fit_transform(X, y)
                    embedding_matrix = torch.from_numpy(projections[name])
                    pids = np.unique(y)
                    nmi, f1, ari, acc, y_pred = evaluate(embedding_matrix, y, n_cluster=len(pids))
                    logger.info(f'nmi: {"%.6f" % nmi}, acc: {"%.6f" % acc}, ari: {"%.6f" % ari}, f1: {"%.6f" % f1}')
    def train_arg(self,X,y,n_neighbors,lr,max_iterations,alpha,dim,prune_ratio,ricci_flow_method,save_nbr_matrix,dataset):
        # train the dataset with different parameters
        logger.info(
            f'neighbors: {n_neighbors} lr: {lr} ricci flow max iterations: {max_iterations} alpha: {alpha} dim: {dim} prune_ratio: {prune_ratio}')
                # rft = RicciFlowTrainer(affinity='rbf')
        rft = RicciFlowTrainer()
        weight_matrix = rft.get_weight_matrix(X, n_neighbors=n_neighbors)

        # get the results of the original weight matrix
        embedding_new = spectral_embedding(
            weight_matrix,
            n_components=dim,
            eigen_solver="arpack",
            eigen_tol="auto",
            random_state=0,
        )
        embedding_matrix = torch.from_numpy(embedding_new)
        pids = np.unique(y)
        nmi, f1, ari, acc, y_pred = evaluate(embedding_matrix, y, n_cluster=len(pids))
        logger.info(
            f'iter time 0, nmi: {"%.6f" % nmi},acc: {"%.6f" % acc},ari: {"%.6f" % ari}, f1: {"%.6f" % f1}')

        # get the results of the updated weight matrix with ricci flow iterations
        if ricci_flow_method == 'Ollivier':
            rft.update_weight_Ollivier(weight_matrix,lr=lr,iterations=max_iterations,alpha=alpha,prune_ratio=prune_ratio,save_nbr_matrix=save_nbr_matrix,dataset_name=dataset)
        elif ricci_flow_method == 'Formann':
            rft.update_weight_Formann(weight_matrix,lr=lr,iterations=max_iterations,prune_ratio=prune_ratio,save_nbr_matrix=save_nbr_matrix,dataset_name=dataset)

        for i in range(max_iterations):
            if ricci_flow_method == 'Ollivier':
                npzfile=np.load(f'./matrixs/{dataset}_Ollivier_lr{lr}_alpha{alpha}_iter{i + 1}_prune_ratio{prune_ratio}.npz')
                weight_matrix_new=csr_matrix(npzfile['matrix'])
            elif ricci_flow_method == 'Formann':
                npzfile=np.load(f'./matrixs/{dataset}_Formann_lr{lr}_iter{i + 1}_prune_ratio{prune_ratio}.npz')
                weight_matrix_new = csr_matrix(npzfile['matrix'])

            embedding_new=spectral_embedding(
                        weight_matrix_new,
                        n_components=dim,
                        eigen_solver="arpack",
                        eigen_tol="auto",
                        random_state=0,
            )

            embedding_matrix=torch.from_numpy(embedding_new)
            pids = np.unique(y)
            nmi, f1, ari, acc,y_pred = evaluate(embedding_matrix, y, n_cluster=len(pids))
            logger.info(f'iter time {i+1}, nmi: {"%.6f" % nmi},acc: {"%.6f" % acc},ari: {"%.6f" % ari}, f1: {"%.6f" % f1}')

    def train_single_dataset(self,dataset, curvature_flow_method='Ollivier'):
        # datasets = ['minist', 'USPS', 'autoUniv_au6_1000','student_prediction', 'User_Knowledge', 'yeast','gender']
        # train for a single dataset
        X, y = Data(self.data_home).load_data(dataset)
        log = f'logs/{dataset}_{curvature_flow_method}.log'
        change_log_file(logger, log)
        logger.info(f'dataset {dataset} load over')
        # config the parameters, you can change the parameters to train the dataset
        # prune_ratio is pruning ratio
        # dim is the target dimension of the embedding
        # lr is the learning rate of the ricci flow (eta)
        # alpha is the alpha parameter of the Ollivier-Ricci flow
        # n_neighbors is the number of neighbors in the affinity matrix
        # max_iterations is the maximum number of iterations of the ricci flow
        prune_ratio_list = [0.0]
        dim_list = [2, 4, 6, 8, 10, 12, 14, 16]
        lr_list = [0.6, 0.7, 0.8, 0.9, 1.0]
        alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5]
        n_neighbors_list = [11]
        max_iterations = [7]
        args = list(product(dim_list, lr_list, n_neighbors_list, alpha_list, prune_ratio_list,max_iterations))
        for arg in args:
            dim = arg[0]
            lr = arg[1]
            n_neighbors = arg[2]
            alpha = arg[3]
            prune_ratio = arg[4]
            max_iterations = arg[5]
            try:
                self.train_arg(X, y, n_neighbors,lr,max_iterations,alpha,dim,prune_ratio, curvature_flow_method, True, dataset)
            except:
                traceback.print_exc()
                continue

if __name__ == '__main__':
    dataset='gender'
    t = Trainer()
    # t.train_baselines(dataset)
    t.train_single_dataset(dataset)

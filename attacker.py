import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from core.gcn import GCN
from core import utils
import os
from multiprocessing import Pool


class BaseAttack(Module):
    """Abstract base class for target attack classes.
    Parameters
    ----------
    model :
        model to attack
    nnodes : int
        number of nodes in the input graph
    device: str
        'cpu' or 'cuda'
    """

    def __init__(self, model, nnodes, device='cpu'):
        super(BaseAttack, self).__init__()
        self.surrogate = model
        self.nnodes = nnodes
        self.device = device

        self.modified_adj = None

    def attack(self, ori_adj, n_perturbations, **kwargs):
        """Generate perturbations on the input graph.
        Parameters
        ----------
        ori_adj : scipy.sparse.csr_matrix
            Original (unperturbed) adjacency matrix.
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
        Returns
        -------
        None.
        """
        raise NotImplementedError()


class RND(BaseAttack):
    def __init__(self, model=None, nnodes=None, device='cpu'):
        super(RND, self).__init__(model, nnodes, device=device)

    def attack(self, ori_features: sp.csr_matrix, ori_adj: sp.csr_matrix, labels: np.ndarray,
               idx_train: np.ndarray, target_node: int, n_perturbations: int, **kwargs):
        """
        Randomly sample nodes u whose label is different from v and
        add the edge u,v to the graph structure. This baseline only
        has access to true class labels in training set
        Parameters
        ----------
        ori_features : scipy.sparse.csr_matrix
            Original (unperturbed) node feature matrix
        ori_adj : scipy.sparse.csr_matrix
            Original (unperturbed) adjacency matrix
        labels :
            node labels
        idx_train :
            node training indices
        target_node : int
            target node index to be attacked
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could be edge removals/additions.
        """

        print(f'number of pertubations: {n_perturbations}')
        modified_adj = ori_adj.tolil()

        row = ori_adj[target_node].todense().A1
        diff_label_nodes = [x for x in idx_train if labels[x] != labels[target_node] and row[x] == 0]
        diff_label_nodes = np.random.permutation(diff_label_nodes)

        if len(diff_label_nodes) >= n_perturbations:
            changed_nodes = diff_label_nodes[: n_perturbations]
            modified_adj[target_node, changed_nodes] = 1
            modified_adj[changed_nodes, target_node] = 1
        else:
            changed_nodes = diff_label_nodes
            unlabeled_nodes = [x for x in range(ori_adj.shape[0]) if x not in idx_train and row[x] == 0]
            unlabeled_nodes = np.random.permutation(unlabeled_nodes)
            changed_nodes = np.concatenate([changed_nodes, unlabeled_nodes[: n_perturbations-len(diff_label_nodes)]])
            modified_adj[target_node, changed_nodes] = 1
            modified_adj[changed_nodes, target_node] = 1
            pass

        self.modified_adj = modified_adj


# TODO: Implemnet your own attacker here
# class MyAttacker(BaseAttack):
#     def __init__(self, model=None, nnodes=None, device='cpu'):
#         super(MyAttacker, self).__init__(model, nnodes, device=device)

#     def attack(self, ori_features, ori_adj, target_node, n_perturbations, **kwargs):
#         pass

class SurrogateModel(GCN):
    def loss(self, features, adj, labels, mask=None, normalize=True):
        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        if normalize:
            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)
        else:
            adj_norm = adj

        output = self.forward(features, adj_norm)
        if mask is not None:
            output = output[mask]
            labels = labels[mask]
        return F.nll_loss(output, labels)


class Nettack(BaseAttack):
    def __init__(self, model=None, nnodes=None, device='cpu'):
        super().__init__(model, nnodes, device)

    def attack(self, ori_features: sp.csr_matrix, ori_adj: sp.csr_matrix, labels: np.ndarray,
               idx_train: np.ndarray, idx_val: np.ndarray, idx_test: np.ndarray,
               target_node: int, n_perturbations: int, **kwargs):
        """
        Implementation of Nettack
        ----------
        ori_features : scipy.sparse.csr_matrix
            Original (unperturbed) node feature matrix
        ori_adj : scipy.sparse.csr_matrix
            Original (unperturbed) adjacency matrix
        labels :
            node labels
        idx_train :
            node training indices
        target_node : int
            target node index to be attacked
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could be edge removals/additions.
        """
        nnodes = ori_adj.shape[0]
        # multithread = True
        multithread = False

        surrogate = GCN(nfeat=ori_features.shape[1], nhid=16,
                        nclass=labels.max().item() + 1, dropout=0.5,
                        with_relu=False, with_bias=False, device=self.device)
        surrogate.fit(ori_features, ori_adj, labels, idx_train, idx_val)

        A = ori_adj.tolil(True)
        W1 = surrogate.gc1.weight.detach().cpu().numpy()
        W2 = surrogate.gc2.weight.detach().cpu().numpy()
        XW = ori_features.dot(W1).dot(W2)

        # candidate_nodes = [x for x in idx_train if x != target_node]
        # candidate_nodes = [x for x in np.concatenate([idx_train, idx_val, idx_test]) if x != target_node]
        # candidate_nodes = [x for x in range(nnodes) if x != target_node]

        # choose top scored nodes as candidates
        if multithread:
            with Pool(processes=os.cpu_count()) as pool:
                scores = pool.map(self.compute_structure_score_mp, [(A, XW, target_node, v, labels[target_node]) for v in range(nnodes) if v != target_node])
        else:
            scores = [(v, self.compute_structure_score(A, XW, target_node, v, labels[target_node])) for v in range(nnodes) if v != target_node]
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        candidate_nodes = [x[0] for x in scores[:300]]

        # Pick n_perturbations nodes to attack from the candidate nodes
        perturbed_nodes = []
        for perturb in range(n_perturbations):
            best_score = float('-inf')
            best_node = None
            if multithread:
                with Pool(processes=os.cpu_count()) as pool:
                    scores = pool.map(self.compute_structure_score_mp, [(A, XW, target_node, v, labels[target_node]) for v in candidate_nodes])
                for v, score in scores:
                    if score > best_score:
                        best_score = score
                        best_node = v
            else:
                for v in candidate_nodes:
                    score = self.compute_structure_score(A, XW, target_node, v, labels[target_node])
                    if score > best_score:
                        best_score = score
                        best_node = v
            if A[target_node, best_node] == 1:
                A[target_node, best_node] = 0
                A[best_node, target_node] = 0
            else:
                A[target_node, best_node] = 1
                A[best_node, target_node] = 1
            candidate_nodes.remove(best_node)
            perturbed_nodes.append(best_node)

            print("Perturbation: {}, best node: {}, best score: {}".format(perturb, best_node, best_score))

            if len(candidate_nodes) == 0:
                print("No candidate nodes left.")
                break

        print("Perturbed nodes: {}".format(perturbed_nodes))
        self.modified_adj = A.tocsr()

        # Verify attack on surrogate model
        # if surrogate.predict(ori_features, A)[target_node].argmax().item() == labels[target_node].item():
        #     print("attack surrogate failed")
        # else:
        #     print("attack surrogate success")

    def compute_structure_score(self, A, XW, target_node, v, c_old):
        if A[target_node, v]:
            A[target_node, v] = 0
            A[v, target_node] = 0
            loss = self.compute_surrogate_loss(A, XW, target_node, c_old)
            A[target_node, v] = 1
            A[v, target_node] = 1
        else:
            A[target_node, v] = 1
            A[v, target_node] = 1
            loss = self.compute_surrogate_loss(A, XW, target_node, c_old)
            A[target_node, v] = 0
            A[v, target_node] = 0
        return loss

    def compute_structure_score_mp(self, args):
        _A, XW, target_node, v, c_old = args
        A = _A.copy()
        if A[target_node, v]:
            A[target_node, v] = 0
            A[v, target_node] = 0
            loss = self.compute_surrogate_loss(A, XW, target_node, c_old)
        else:
            A[target_node, v] = 1
            A[v, target_node] = 1
            loss = self.compute_surrogate_loss(A, XW, target_node, c_old)
        return v, loss

    def compute_surrogate_loss(self, A, XW, target_node, c_old):
        A_hat = self.compute_A_hat(A)
        A_hat_square = A_hat.dot(A_hat)
        logits = A_hat_square.dot(XW)
        if c_old == logits[target_node].argsort()[-1]:
            c_new = logits[target_node].argsort()[-2]
        else:
            c_new = logits[target_node].argsort()[-1]
        return logits[target_node, c_new] - logits[target_node, c_old]

    def compute_A_hat(self, A):
        A_wave = A + sp.eye(A.shape[0])
        D_wave = A_wave.sum(1).A1
        D_wave_inv_sqrt = sp.diags(np.power(D_wave, -0.5))
        A_hat = D_wave_inv_sqrt.dot(A_wave).dot(D_wave_inv_sqrt)
        return A_hat

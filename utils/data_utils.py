import numpy as np
import random as rd
import scipy.sparse as sp
import os
from random import shuffle,randint,choice,sample
from logzero import logger
import torch
import pickle
import obonet
import networkx as nx

root_terms = {'mf': 'GO:0003674', 'bp': 'GO:0008150', 'cc': 'GO:0005575'}
class Data(object):
    def __init__(self, path, NS, batch_size):
        self.path = path
        self.batch_size = batch_size
        self.NS = NS

        train_file = f'{path}/{NS}/train.txt'
        test_file = f'{path}/{NS}/test.txt'

        #get number of users and items
        self.n_proteins, self.n_terms = 0, 0
        self.n_train, self.n_test = 0, 0

        self.exist_proteins = []
        # get number of proteins and terms from training set
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    terms = [int(l[1])]
                    uid = int(l[0])
                    self.exist_proteins.append(uid)
                    self.n_terms = max(self.n_terms, max(terms))
                    self.n_proteins = max(self.n_proteins, uid)
                    self.n_train += len(terms)
        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    # try:
                    #     terms = [int(i) for i in l.split(' ')[1:]]
                    # except Exception:
                    terms = [int(l.split(' ')[1])]
                        # continue
                    self.n_terms = max(self.n_terms, max(terms))
                    self.n_test += len(terms)
        
        self.n_terms += 1
        self.n_proteins += 1
        logger.info('n_proteins=%d, n_terms=%d' % (self.n_proteins, self.n_terms))
        logger.info('n_interactions=%d' % (self.n_train + self.n_test))
        logger.info('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_proteins * self.n_terms)))

        self.R = sp.dok_matrix((self.n_proteins, self.n_terms), dtype=np.float32) # 用字典的方式获取蛋白质-GO的标注矩阵
        
        self.train_terms, self.test_set = {}, {}
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0:
                        break
                    l = l.strip('\n')
                    terms = [i for i in l.split(' ')]
                    uid, train_uid_term = int(terms[0]), int(terms[1])
                    try:
                        score = float(terms[2])
                    except Exception:
                        score = 1.0
                    if uid not in self.train_terms:
                        self.train_terms[uid] = {}
                    self.train_terms[uid][train_uid_term] = score

                for uid in self.train_terms:
                    for i in self.train_terms[uid]:
                        self.R[uid, i] = self.train_terms[uid][i]
                        # R[uid][i] = 1

                    # train_terms[uid] = train_uid_terms

                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    try:
                        terms = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue

                    uid, test_uid_term = terms[0], terms[1]
                    if uid not in self.test_set:
                        self.test_set[uid] = []
                    self.test_set[uid].append(test_uid_term)

        self.training_data = []
        for uid in self.train_terms:
            for term in self.train_terms[uid]:
                self.training_data.append((uid, term))
    
    def next_batch_pairwise(self, n_negs=1):
        shuffle(self.training_data)
        ptr = 0
        data_size = len(self.training_data)
        while ptr < data_size:
            if ptr + self.batch_size < data_size:
                batch_end = ptr + self.batch_size
            else:
                batch_end = data_size
            proteins = [self.training_data[idx][0] for idx in range(ptr, batch_end)]
            terms = [self.training_data[idx][1] for idx in range(ptr, batch_end)]
            ptr = batch_end
            u_idx, i_idx, j_idx = [], [], []
            term_list = list(range(self.n_terms))
            for i, protein in enumerate(proteins):
                i_idx.append(terms[i])
                u_idx.append(protein)
                for m in range(n_negs):
                    neg_item = choice(term_list)
                    while neg_item in self.train_terms[protein]:
                        neg_item = choice(term_list)
                    j_idx.append(neg_item)
            yield u_idx, i_idx, j_idx

    def create_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_proteins + self.n_terms, self.n_proteins + self.n_terms), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = R.tolil()

        adj_mat[:self.n_proteins, self.n_proteins:] = R   # 前n_proteins行是蛋白质节点，后n_terms行是go terms节点，矩阵的右上角为标注矩阵
        adj_mat[self.n_proteins:, :self.n_proteins] = R.T  # 前n_proteins列是蛋白质节点，后n_terms列是go terms节点，矩阵的左下角为标注矩阵的转置
        adj_mat = adj_mat.todok()

        def mean_adj_single(adj):
            # D^-1 * A
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            logger.info('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()
        norm_adj_mat = mean_adj_single(adj_mat + sp.eye(adj_mat.shape[0])).tocsr()
        return norm_adj_mat
    
    def load_ppi_mat(self):
        ppi_mat_path = f'{self.path}/{self.NS}/ppi_mat.npz'
        if os.path.exists(ppi_mat_path):
            ppi_mat = sp.load_npz(ppi_mat_path)
        else:
            protein2idx = {}
            with open(f'{self.path}/protein_list.txt', 'r') as f:
                # f.readline()
                for line in f:
                    protein, idx = line.strip().split(' ')
                    protein2idx[protein] = int(idx)
            ppi_mat = sp.dok_matrix((self.n_proteins, self.n_proteins), dtype=np.float32) # 用字典的方式获取ppi矩阵
            with open(f'{self.path}/network.txt', 'r') as f:
                for line in f:
                    protein1, protein2, score = line.strip().split(' ')
                    pid1 = protein2idx.get(protein1, -1)
                    pid2 = protein2idx.get(protein2, -1)
                    if pid1!= -1 and pid2!= -1:
                        ppi_mat[pid1, pid2] = float(score)
                        ppi_mat[pid2, pid1] = float(score)
            #加入self-loop
            for i in range(self.n_proteins):
                ppi_mat[i,i] = 1.0

            # 保存矩阵
            ppi_mat = ppi_mat.tocsr()
            sp.save_npz(ppi_mat_path, ppi_mat)
        protein_embedding_dic = pickle.load(open(f'{self.path}/protein_embedding.pkl', 'rb'))
        protein_embedding = [protein_embedding_dic[uid] for uid, pid in sorted(protein2idx.items(), key=lambda x:x[1])]
        protein_embedding = torch.vstack(protein_embedding)
        return ppi_mat, protein_embedding
    
    def load_go_mat(self):
        go_mat_path = f'{self.path}/{self.NS}/go_mat.npz'
        go2idx = {}
        with open(f'{self.path}/{self.NS}/term_list.txt', 'r') as f:
            # f.readline()
            for line in f:
                go_name, gid = line.strip().split(' ')
                go2idx[go_name] = int(gid)
        if os.path.exists(go_mat_path):
            go_mat = sp.load_npz(go_mat_path)
        else:
            go_graph = obonet.read_obo(open(f'{self.path}/go-basic.obo', 'r'))
            accepted_edges = set()
            unaccepted_edges = set()

            for edge in go_graph.edges:
                if edge[2] == 'is_a' or edge[2] == 'part_of':
                    accepted_edges.add(edge)
                else:
                    unaccepted_edges.add(edge)
            go_graph.remove_edges_from(unaccepted_edges)

            go_mat = sp.dok_matrix((len(go2idx), len(go2idx)), dtype=np.float32)
            for node in go_graph.nodes:
                if node in go2idx:
                    idx = go2idx[node]
                    for node2 in nx.descendants(go_graph, node).union(set([node, ])):
                        idx2 = go2idx.get(node2, -1)
                        if idx2!= -1:
                            go_mat[idx2, idx] = 1.0
            go_mat = go_mat.tocsr()
            sp.save_npz(go_mat_path, go_mat)

        # PubMedBERT embeddings are used for GO term embedding
        go_embedding_dic = pickle.load(open(f'{self.path}/go_term_embedding_path', 'rb'))
        go_embedding = [go_embedding_dic[go_name] if go_name in go_embedding_dic else go_embedding_dic[root_terms[self.NS]] for go_name, gid in sorted(go2idx.items(), key=lambda x:x[1])]
        go_embedding = torch.from_numpy(np.stack(go_embedding, axis=0))
        return go_mat, go_embedding

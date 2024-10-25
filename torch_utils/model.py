import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import norm


class GraphConv(nn.Module):
    """
    The graph convolutional layer.
    """
    def __init__(self, in_features, out_features):
        super(GraphConv, self).__init__()
        self.lin = nn.Linear(in_features, out_features)
    
    def forward(self, x, A):
        """
        :param x: n_nodes x in_features, input features
        :param A: n_nodes x n_nodes, adjacency matrix
        :return: n_nodes x out_features, output features
        """
        x = self.lin(x)
        # x = torch.sparse.mm(A, x)
        return A.matmul(x)


class DualGOFiller(nn.Module):
    '''
    bipartite graph channel
    PPI/DAG graph channel
    '''
    def __init__(self, pro_num, term_num, esm_dim, pub_dim, hidden_dim, bi_layers=2):
        '''
        :param pro_num: the number of proteins
        :param term_num: the number of terms
        :param esm_dim: the dimension of ESM1b embeddings
        :param pub_dim: the dimension of PubMedBERT embeddings
        :param hidden_dim: the dimension for protein and term embeddings in biparite graph
        :param bi_layers: the number of layers for bipartite graph
        '''
        super(DualGOFiller, self).__init__()
        self.pro_num = pro_num
        self.term_num = term_num
        self.hidden_dim = hidden_dim
        self.bi_layers = bi_layers
        self.dense_esm0 = nn.Linear(esm_dim, 512)
        self.dense_pub0 = nn.Linear(pub_dim, 512)
        self.bm_esm0 = norm.BatchNorm(512)
        self.bm_pub0 = norm.BatchNorm(512)

        self.gcn_esm1 = GraphConv(512, 256)
        self.bm_esm1 = norm.BatchNorm(256)
        self.gcn_esm2 = GraphConv(256, (bi_layers+1) * hidden_dim)
        self.bm_esm2 = norm.BatchNorm((bi_layers+1) * hidden_dim)
        self.dense_esm1 = nn.Linear((bi_layers+1) * hidden_dim, (bi_layers+1) * hidden_dim)
        self.bm_esm3 = norm.BatchNorm((bi_layers+1) * hidden_dim)
        self.dense_esm2 = nn.Linear((bi_layers+1) * hidden_dim, (bi_layers+1) * hidden_dim)
        self.bm_esm4 = norm.BatchNorm((bi_layers+1) * hidden_dim)
        
        self.gcn_pub1 = GraphConv(512, 256)
        self.bm_pub1 = norm.BatchNorm(256)
        self.gcn_pub2 = GraphConv(256, (bi_layers+1) * hidden_dim)
        self.bm_pub2 = norm.BatchNorm((bi_layers+1) * hidden_dim)
        self.dense_pub1 = nn.Linear((bi_layers+1) * hidden_dim, (bi_layers+1) * hidden_dim)
        self.bm_pub3 = norm.BatchNorm((bi_layers+1) * hidden_dim)
        self.dense_pub2 = nn.Linear((bi_layers+1) * hidden_dim, (bi_layers+1) * hidden_dim)
        self.bm_pub4 = norm.BatchNorm((bi_layers+1) * hidden_dim)

        self.embedding_dict, self.weight_dict = self._init_model()
        # self.bi_gcn1 = GraphConv(hidden_dim, hidden_dim)
        # self.bi_gcn2 = GraphConv(hidden_dim, hidden_dim)
        # self.bm_bi1 = norm.BatchNorm(hidden_dim)
        # self.bm_bi2 = norm.BatchNorm(hidden_dim)
    
    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'pro_emb': nn.Parameter(initializer(torch.empty(self.pro_num, self.hidden_dim))),
            'term_emb': nn.Parameter(initializer(torch.empty(self.term_num, self.hidden_dim))),
        })

        weight_dict = nn.ParameterDict()
        layers = [self.hidden_dim] + self.bi_layers * [self.hidden_dim]
        for k in range(self.bi_layers):
            weight_dict.update({'W_gc_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_gc_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

            weight_dict.update({'W_bi_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_bi_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

        return embedding_dict, weight_dict
    
    def forward(self, X_esm, X_pub, A_esm, A_pub, A_bi):
        """
        :param X_esm: n_proteins x esm_dim, ESM1b embeddings for proteins
        :param X_pub: n_terms x pub_dim, PubMedBERT embeddings for terms
        :param A_esm: n_proteins x n_proteins, PPI/DAG adjacency matrix for proteins
        :param A_pub: n_terms x n_terms, PPI/DAG adjacency matrix for terms
        :param A_bi: n_proteins + n_terms x n_proteins + n_terms, bipartite graph adjacency matrix
        :return: pro_emb1, term_emb1, pro_emb2, term_emb2
        """
        # ESM1b embeddings
        X_esm = F.leaky_relu(self.dense_esm0(X_esm))
        X_esm = self.bm_esm0(X_esm)
        X_esm = F.leaky_relu(self.gcn_esm1(X_esm, A_esm))
        X_esm = self.bm_esm1(X_esm)
        X_esm = F.leaky_relu(self.gcn_esm2(X_esm, A_esm))
        X_esm = self.bm_esm2(X_esm)
        X_esm = F.leaky_relu(self.dense_esm1(X_esm))
        X_esm = self.bm_esm3(X_esm)
        X_esm = F.leaky_relu(self.dense_esm2(X_esm))
        X_esm = self.bm_esm4(X_esm)

        # PubMedBERT embeddings
        X_pub = F.leaky_relu(self.dense_pub0(X_pub))
        X_pub = self.bm_pub0(X_pub)
        X_pub = F.leaky_relu(self.gcn_pub1(X_pub, A_pub))
        X_pub = self.bm_pub1(X_pub)
        X_pub = F.leaky_relu(self.gcn_pub2(X_pub, A_pub))
        X_pub = self.bm_pub2(X_pub)
        X_pub = F.leaky_relu(self.dense_pub1(X_pub))
        X_pub = self.bm_pub3(X_pub)
        X_pub = F.leaky_relu(self.dense_pub2(X_pub))
        X_pub = self.bm_pub4(X_pub)

        # Biparite graph embeddings
        ego_embeddings = torch.cat([self.embedding_dict['pro_emb'], self.embedding_dict['term_emb']], 0)
        all_embeddings = [ego_embeddings]
        # for k in range(self.layers):
        #     ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
        #     all_embeddings += [ego_embeddings]
        for k in range(self.bi_layers):
            side_embeddings = torch.sparse.mm(A_bi, ego_embeddings)

            # transformed sum messages of neighbors.
            sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k]) \
                                             + self.weight_dict['b_gc_%d' % k]

            # bi messages of neighbors.
            # element-wise product
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            # transformed bi messages of neighbors.
            bi_embeddings = torch.matmul(bi_embeddings, self.weight_dict['W_bi_%d' % k]) \
                                            + self.weight_dict['b_bi_%d' % k]

            # non-linear activation.
            ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(sum_embeddings + bi_embeddings)

            # message dropout.
            ego_embeddings = nn.Dropout(0.1)(ego_embeddings)

            # normalize the distribution of embeddings.
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)

            all_embeddings += [norm_embeddings]
        # all_embeddings = torch.stack(all_embeddings, dim=1)
        # all_embeddings = torch.mean(all_embeddings, dim=1)
        all_embeddings = torch.cat(all_embeddings, dim=1)
        pro_emb = all_embeddings[:self.pro_num]
        term_emb = all_embeddings[self.pro_num:]

        return X_esm, X_pub, pro_emb, term_emb

import os
import math
import networkx as nx
import numpy as np
import scipy as sp
import scipy.sparse
import torch
import torch.nn.functional as F
# import community as community_louvain
from sklearn.metrics import mean_absolute_error

import pandas as pd
from extract_feats import extract_feats, extract_numbers
import networkx as nx

from torch import Tensor
from torch.utils.data import Dataset

from grakel.utils import graph_from_networkx
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from tqdm import tqdm
import scipy.sparse as sparse
from torch_geometric.data import Data

from extract_feats import extract_feats, extract_numbers
from node2vec import Node2Vec

from networkx.algorithms import community



def preprocess_dataset(dataset, n_max_nodes, spectral_emb_dim):

    data_lst = []
    if dataset == 'test':
        filename = './data/dataset_'+dataset+'.pt'
        desc_file = './data/'+dataset+'/test.txt'

        if os.path.isfile(filename):
            data_lst = torch.load(filename)
            print(f'Dataset {filename} loaded from file')

        else:
            fr = open(desc_file, "r")
            for line in fr:
                line = line.strip()
                tokens = line.split(",")
                graph_id = tokens[0]
                desc = tokens[1:]
                desc = "".join(desc)
                feats_stats = extract_numbers(desc)
                feats_stats = torch.FloatTensor(feats_stats).unsqueeze(0)
                data_lst.append(Data(stats=feats_stats, filename = graph_id))
            fr.close()
            torch.save(data_lst, filename)
            print(f'Dataset {filename} saved')


    else:
        filename = './data/dataset_'+dataset+'.pt'
        graph_path = './data/'+dataset+'/graph'
        desc_path = './data/'+dataset+'/description'

        if os.path.isfile(filename):
            data_lst = torch.load(filename)
            print(f'Dataset {filename} loaded from file')

        else:
            # traverse through all the graphs of the folder
            files = [f for f in os.listdir(graph_path)]
            adjs = []
            eigvals = []
            eigvecs = []
            n_nodes = []
            max_eigval = 0
            min_eigval = 0
            for fileread in tqdm(files):
                tokens = fileread.split("/")
                idx = tokens[-1].find(".")
                filen = tokens[-1][:idx]
                extension = tokens[-1][idx+1:]
                fread = os.path.join(graph_path,fileread)
                fstats = os.path.join(desc_path,filen+".txt")
                #load dataset to networkx
                if extension=="graphml":
                    G = nx.read_graphml(fread)
                    # Convert node labels back to tuples since GraphML stores them as strings
                    G = nx.convert_node_labels_to_integers(
                        G, ordering="sorted"
                    )
                else:
                    G = nx.read_edgelist(fread)
                # use canonical order (BFS) to create adjacency matrix
                ### BFS & DFS from largest-degree node

                
                CGs = [G.subgraph(c) for c in nx.connected_components(G)]

                # rank connected componets from large to small size
                CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)

                node_list_bfs = []
                for ii in range(len(CGs)):
                    node_degree_list = [(n, d) for n, d in CGs[ii].degree()]
                    degree_sequence = sorted(
                    node_degree_list, key=lambda tt: tt[1], reverse=True)

                    bfs_tree = nx.bfs_tree(CGs[ii], source=degree_sequence[0][0])
                    node_list_bfs += list(bfs_tree.nodes())

                adj_bfs = nx.to_numpy_array(G, nodelist=node_list_bfs)

                adj = torch.from_numpy(adj_bfs).float()
                edge_index = torch.nonzero(adj).t()

                size_diff = n_max_nodes - G.number_of_nodes()
                diags = np.sum(adj_bfs, axis=0)
                diags = np.squeeze(np.asarray(diags))
                D = sparse.diags(diags).toarray()
                L = D - adj_bfs
                with np.errstate(divide="ignore"):
                    diags_sqrt = 1.0 / np.sqrt(diags)
                diags_sqrt[np.isinf(diags_sqrt)] = 0
                DH = sparse.diags(diags).toarray()
                L = np.linalg.multi_dot((DH, L, DH))
                L = torch.from_numpy(L).float()
                eigval, eigvecs = torch.linalg.eigh(L)
                eigval = torch.real(eigval)
                eigvecs = torch.real(eigvecs)
                idx = torch.argsort(eigval)
                eigvecs = eigvecs[:,idx]
                
                x = torch.zeros(G.number_of_nodes(), spectral_emb_dim+1)
                x[:,0] = torch.mm(adj, torch.ones(G.number_of_nodes(), 1))[:,0]/(n_max_nodes-1)
                mn = min(G.number_of_nodes(),spectral_emb_dim)
                mn+=1
                x[:,1:mn] = eigvecs[:,:spectral_emb_dim]
                adj = F.pad(adj, [0, size_diff, 0, size_diff])
                adj = adj.unsqueeze(0)

                feats_stats = extract_feats(fstats)
                feats_stats = torch.FloatTensor(feats_stats).unsqueeze(0)

                data_lst.append(Data(x=x, edge_index=edge_index, A=adj, stats=feats_stats, filename = filen))
            torch.save(data_lst, filename)
            print(f'Dataset {filename} saved')
    return data_lst


        

def construct_nx_from_adj(adj):
    G = nx.from_numpy_array(adj, create_using=nx.Graph)
    to_remove = []
    for node in G.nodes():
        if G.degree(node) == 0:
            to_remove.append(node)
    G.remove_nodes_from(to_remove)
    return G



def handle_nan(x):
    if math.isnan(x):
        return float(-100)
    return x




def masked_instance_norm2D(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-5):
    """
    x: [batch_size (N), num_objects (L), num_objects (L), features(C)]
    mask: [batch_size (N), num_objects (L), num_objects (L), 1]
    """
    mask = mask.view(x.size(0), x.size(1), x.size(2), 1).expand_as(x)
    mean = (torch.sum(x * mask, dim=[1,2]) / torch.sum(mask, dim=[1,2]))   # (N,C)
    var_term = ((x - mean.unsqueeze(1).unsqueeze(1).expand_as(x)) * mask)**2  # (N,L,L,C)
    var = (torch.sum(var_term, dim=[1,2]) / torch.sum(mask, dim=[1,2]))  # (N,C)
    mean = mean.unsqueeze(1).unsqueeze(1).expand_as(x)  # (N, L, L, C)
    var = var.unsqueeze(1).unsqueeze(1).expand_as(x)    # (N, L, L, C)
    instance_norm = (x - mean) / torch.sqrt(var + eps)   # (N, L, L, C)
    instance_norm = instance_norm * mask
    return instance_norm


def masked_layer_norm2D(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-5):
    """
    x: [batch_size (N), num_objects (L), num_objects (L), features(C)]
    mask: [batch_size (N), num_objects (L), num_objects (L), 1]
    """
    mask = mask.view(x.size(0), x.size(1), x.size(2), 1).expand_as(x)
    mean = torch.sum(x * mask, dim=[3,2,1]) / torch.sum(mask, dim=[3,2,1])   # (N)
    var_term = ((x - mean.view(-1,1,1,1).expand_as(x)) * mask)**2  # (N,L,L,C)
    var = (torch.sum(var_term, dim=[3,2,1]) / torch.sum(mask, dim=[3,2,1]))  # (N)
    mean = mean.view(-1,1,1,1).expand_as(x)  # (N, L, L, C)
    var = var.view(-1,1,1,1).expand_as(x)    # (N, L, L, C)
    layer_norm = (x - mean) / torch.sqrt(var + eps)   # (N, L, L, C)
    layer_norm = layer_norm * mask
    return layer_norm


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

def graph_features_error(adj, cond):
    adj = adj.detach().cpu().numpy()
    G = nx.from_numpy_array(adj)
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    avg_degree = sum(dict(G.degree()).values()) / num_nodes if num_nodes > 0 else 0
    triangles = sum(nx.triangles(G).values()) // 3
    clustering_coefficient = nx.average_clustering(G)
    k_core = max(nx.core_number(G).values())
    num_communities = len(list(nx.community.greedy_modularity_communities(G)))

    generated_features = [num_nodes, num_edges, avg_degree, triangles, clustering_coefficient, k_core, num_communities]
    return mean_absolute_error(generated_features, cond)





def extract_graph_properties(adj):
    G = construct_nx_from_adj(adj)
    nb_of_nodes = G.number_of_nodes()
    nb_of_edges = G.number_of_edges()
    avg_degree = 2 * nb_of_edges / nb_of_nodes if nb_of_nodes > 0 else 0
    triangles = sum(nx.triangles(G).values()) // 3
    clustering_coefficient = nx.transitivity(G)
    k_core = max(nx.core_number(G).values()) if nb_of_nodes > 0 else 0
    communities = len(list(community.greedy_modularity_communities(G)))
    
    properties = np.array([nb_of_nodes, nb_of_edges ,avg_degree,triangles,clustering_coefficient,k_core,communities])
    return G.edges(), properties



def normalization(prop):
    coefs = [50, 2000, 50, 20000, 1, 50, 20]
    return prop/coefs



def features_error(adj, cond):

    adj_soft = torch.sigmoid(adj * 10) 
    node_activity = torch.sum(adj_soft, dim=1)
    active_nodes = torch.sigmoid(node_activity * 10) 
    num_nodes = torch.sum(active_nodes) 
    
    num_edges = torch.sum(adj_soft) / 2 
    avg_degree = torch.sum(adj_soft) / (num_nodes + 1e-6)
    triangles = torch.sum(torch.diagonal(torch.matmul(torch.matmul(adj_soft, adj_soft), adj_soft))) / 6
    
    degrees = torch.sum(adj_soft, dim=1)
    possible_triangles = degrees * (degrees - 1) + 1e-6 
    
    triangles_per_node = torch.diagonal(torch.matmul(torch.matmul(adj_soft, adj_soft), adj_soft)) / 2
    clustering_coeffs = triangles_per_node / possible_triangles
    clustering_coefficient = torch.sum(clustering_coeffs * active_nodes) / (torch.sum(active_nodes) + 1e-6)
    
    generated_features = torch.stack([
        num_nodes,
        num_edges,
        avg_degree,
        triangles,
        clustering_coefficient,
        torch.mean(degrees * active_nodes)
    ])

    generated_features = F.normalize(generated_features, p=2, dim=0)
    cond = F.normalize(cond[:-1].to(torch.float32), p=2, dim=0)
    
    loss = torch.mean(torch.abs(generated_features - cond))

    return loss

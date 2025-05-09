import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from torch_geometric.nn import GINConv, GATConv, GATv2Conv, GCNConv, TransformerConv, ResGatedGraphConv, SAGEConv
from torch_geometric.nn import global_add_pool

    
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes, d_cond = 7):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes

        mlp_layers = [nn.Linear(latent_dim + d_cond, hidden_dim)] 
        bn_layers = [nn.BatchNorm1d(hidden_dim)]

        for i in range(1, n_layers) : 
            mlp_layers.append(nn.Linear(hidden_dim*i + d_cond, hidden_dim*(i+1)))
            bn_layers.append(nn.BatchNorm1d(hidden_dim*(i+1)))
  
        mlp_layers.append(nn.Linear(hidden_dim*n_layers, 2*n_nodes*(n_nodes-1)//2))
        bn_layers.append(nn.BatchNorm1d(hidden_dim*n_layers))

        self.mlp = nn.ModuleList(mlp_layers)
        self.bn = nn.ModuleList(bn_layers)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, cond, temp):
        for i in range(self.n_layers):
            x = torch.cat((x, cond), dim=1)
            x = self.relu(self.mlp[i](x))
            x = self.bn[i](x)

        x = self.mlp[self.n_layers](x)

        x = torch.reshape(x, (x.size(0), -1, 2))
        x = F.gumbel_softmax(x, tau=temp, hard=True)[:,:,0]

        adj = torch.zeros(x.size(0), self.n_nodes, self.n_nodes, device=x.device)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        adj[:,idx[0],idx[1]] = x
        adj = adj + torch.transpose(adj, 1, 2)
        return adj


class GAT(torch.nn.Module): 
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim // 4, heads=4, concat=True))     

        for _ in range(n_layers-1):
            self.convs.append(GATConv(hidden_dim, hidden_dim // 4, heads=4, concat=True))
        
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),  
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim),  
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(hidden_dim),
                            nn.Linear(hidden_dim, latent_dim), 
                            nn.LeakyReLU(0.2))
        
        self.bn = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for i in range(n_layers)])
        

    def forward(self, data):
        edge_index = data.edge_index
        x = data.x

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)  
            x = self.bn[i](x) 
            x = F.relu(x)  
            x = F.dropout(x, p=self.dropout, training=self.training)

        out = global_add_pool(x, data.batch)
        out = self.mlp(out)
        return out
    

class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim),  
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim), 
                            nn.LeakyReLU(0.2))
                            ))                        
        for layer in range(n_layers-1):
            self.convs.append(GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim),  
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim), 
                            nn.LeakyReLU(0.2))
                            )) 

        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, latent_dim*2)

    def forward(self, data):
        edge_index = data.edge_index
        x = data.x

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(x, self.dropout, training=self.training)

        out = global_add_pool(x, data.batch)
        out = self.bn(out)
        out = self.fc(out)
        return out
    
class SAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2): 
        super().__init__()
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))  
        for _ in range(n_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))  

        self.bn = nn.BatchNorm1d(hidden_dim)  
        self.fc = nn.Linear(hidden_dim, latent_dim) 

        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),  
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim),  
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(hidden_dim),
                            nn.Linear(hidden_dim, latent_dim), 
                            nn.LeakyReLU(0.2))
        
    def forward(self, data):
        edge_index = data.edge_index
        x = data.x

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x) 
            x = F.dropout(x, self.dropout, training=self.training) 

        out = global_add_pool(x, data.batch)  
        out = self.bn(out) 
        out = self.mlp(out)  
        return out


# Variational Autoencoder
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec, n_max_nodes, current_temp = 1.0):
        super(VariationalAutoEncoder, self).__init__()
        self.n_max_nodes = n_max_nodes
        self.input_dim = input_dim
        self.encoder = SAGE(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc)
        self.fc_mu = nn.Sequential(nn.Linear(hidden_dim_enc, latent_dim))
        self.fc_logvar = nn.Sequential(nn.Linear(hidden_dim_enc, latent_dim))
        self.decoder = Decoder(latent_dim, hidden_dim_dec, n_layers_dec, n_max_nodes)
        self.current_temp = current_temp
        self.decay_rate = 0.99

    def forward(self, data, cond):
        x_g = self.encoder(data)
        x_g =  x_g
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g, cond, self.current_temp)
        return adj

    def update_temperature(self):
        self.current_temp = max(0.1, self.current_temp * self.decay_rate)
        return self.current_temp
    
    def encode(self, data):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        return x_g

    def reparameterize(self, mu, logvar, eps_scale=1.):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, mu, logvar, cond):
       x_g = self.reparameterize(mu, logvar)
       adj = self.decoder(x_g, cond, self.current_temp)
       return adj

    def decode_mu(self, mu, cond):
       adj = self.decoder(mu, cond, 0.90)
       return adj

    def loss_function(self, data, cond, beta=0.02):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g, cond, self.current_temp)
        recon = F.l1_loss(adj, data.A, reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon + beta*kld

        return loss, recon, kld

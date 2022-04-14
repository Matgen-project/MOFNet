from ast import mod
from turtle import forward
from .egnn import *
from .painn import *
from .schnet import *
from .dimenet_pp import *
from torch import nn
from torch.nn import functional as F

def make_baseline_model(d_atom, model_name, N=2, d_model=128, use_global_feature=False, d_feature=9, **kwargs):
    model = None
    if model_name == 'egnn':
        representation = EGNN(in_node_nf=d_atom, hidden_nf=d_model, n_layers=N, attention=True)
        use_adj = True
    elif model_name == 'dimenetpp':
        representation = DimeNetPlusPlus(hidden_channels=d_model, out_channels=d_model, num_input=d_atom, num_blocks=N, int_emb_size=d_model // 2, basis_emb_size=8, out_emb_channels=d_model * 2, num_spherical=7, num_radial=6)
        use_adj = True
    elif model_name == 'schnet':
        representation = SchNet(n_atom_basis=d_model, n_filters=d_model, n_interactions=N, max_z=d_atom)
        use_adj = False
    elif model_name == 'painn':
        representation = PaiNN(n_atom_basis=d_model, n_interactions=N, max_z=d_atom)
        use_adj = False
    if use_global_feature:
        out = Generator_with_gf(d_model=d_model, d_gf=d_feature)
    else:
        out = Generator(d_model=d_model)
    model = BaselineModel(representation=representation, output=out, use_adj=use_adj)
    return model

class Generator(nn.Module):
    def __init__(self, d_model):
        super(Generator, self).__init__()
        self.hidden_nf = d_model
        self.node_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                      nn.SiLU(),
                                      nn.Linear(self.hidden_nf, self.hidden_nf))

        self.graph_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                       nn.SiLU(),
                                       nn.Linear(self.hidden_nf, 1))

    def forward(self, h, atom_mask, global_feature=None):
        h = self.node_dec(h)
        h = h * atom_mask.unsqueeze(-1)
        h = torch.sum(h, dim=1)
        pred = self.graph_dec(h)
        return pred.squeeze(1)  

class Generator_with_gf(nn.Module):
    def __init__(self, d_model, d_gf):
        super(Generator_with_gf, self).__init__()
        self.hidden_nf = d_model
        self.input_nf = d_gf
        self.node_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                      nn.SiLU(),
                                      nn.Linear(self.hidden_nf, self.hidden_nf))

        self.gf_enc = nn.Sequential(nn.Linear(self.input_nf, self.hidden_nf // 2),
                                      nn.SiLU(),
                                      nn.Linear(self.hidden_nf // 2, self.hidden_nf))

        self.graph_dec = nn.Sequential(nn.Linear(self.hidden_nf * 2, self.hidden_nf),
                                       nn.SiLU(),
                                       nn.Linear(self.hidden_nf, 1))

    def forward(self, h, atom_mask, global_feature):
        h = self.node_dec(h)
        h = h * atom_mask.unsqueeze(-1)
        h = torch.sum(h, dim=1)
        g = self.gf_enc(global_feature)
        h = torch.cat([h,g], dim=1)
        pred = self.graph_dec(h)
        return pred.squeeze(1)

class BaselineModel(nn.Module):
    def __init__(self, representation, output, use_adj=True):
        super(BaselineModel, self).__init__()
        self.representation = representation
        self.output = output
        self.use_adj = use_adj
    def forward(self, node_features, batch_mask, pos, adj, global_feature=None):
        if not self.use_adj:
            neighbors, neighbor_mask = adj
            rep = self.representation(node_features, pos, neighbors, neighbor_mask, batch_mask)
        else:
            rep = self.representation(node_features, batch_mask, pos, adj)
        out = self.output(rep, batch_mask, global_feature)
        return out
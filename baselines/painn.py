import math
from . import spk_utils as snn
import torch
import torch.nn as nn
import torch.nn.functional as F
from .spk_utils.neighbors import atom_distances
from typing import Union, Callable

class BesselBasis(nn.Module):
    """
    Sine for radial basis expansion with coulomb decay. (0th order Bessel from DimeNet)
    """

    def __init__(self, cutoff=5.0, n_rbf=None):
        """
        Args:
            cutoff: radial cutoff
            n_rbf: number of basis functions.
        """
        super(BesselBasis, self).__init__()
        # compute offset and width of Gaussian functions
        freqs = torch.arange(1, n_rbf + 1) * math.pi / cutoff
        self.register_buffer("freqs", freqs)

    def forward(self, inputs):
        a = self.freqs[None, None, None, :]
        ax = inputs * a
        sinax = torch.sin(ax)

        norm = torch.where(inputs == 0, torch.tensor(1.0, device=inputs.device), inputs)
        y = sinax / norm

        return y

act_class_mapping = {
    "ssp": snn.ShiftedSoftplus,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


class GatedEquivariantBlock(nn.Module):
    """Gated Equivariant Block as defined in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties and molecular spectra
    """

    def __init__(
        self,
        hidden_channels,
        out_channels,
        intermediate_channels=None,
        activation="silu",
        scalar_activation=False,
    ):
        super(GatedEquivariantBlock, self).__init__()
        self.out_channels = out_channels

        if intermediate_channels is None:
            intermediate_channels = hidden_channels

        self.vec1_proj = nn.Linear(hidden_channels, hidden_channels)
        self.vec2_proj = nn.Linear(hidden_channels, out_channels)

        act_class = act_class_mapping[activation]
        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, intermediate_channels),
            act_class(),
            nn.Linear(intermediate_channels, out_channels * 2),
        )

        self.act = act_class() if scalar_activation else None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)

    def forward(self, x, v):
        vec1 = torch.norm(self.vec1_proj(v), dim=-2)
        vec2 = self.vec2_proj(v)

        x = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(2) * vec2

        if self.act is not None:
            x = self.act(x)
        return x, v

class PaiNN(nn.Module):
    """ Polarizable atom interaction neural network """
    def __init__(
            self,
            n_atom_basis: int = 128,
            n_interactions: int = 3,
            n_rbf: int = 20,
            cutoff: float = 5.,
            cutoff_network: Union[nn.Module, str] = 'cosine',
            radial_basis: Callable = BesselBasis,
            activation=F.silu,
            max_z: int = 100,
            store_neighbors: bool = False,
            store_embeddings: bool = False,
            n_edge_features: int = 0,
    ):
        super(PaiNN, self).__init__()

        self.n_atom_basis = n_atom_basis
        self.n_interactions = n_interactions
        self.cutoff = cutoff
        self.cutoff_network = snn.get_cutoff_by_string(cutoff_network)(cutoff)
        self.radial_basis = radial_basis(cutoff=cutoff, n_rbf=n_rbf)
        self.embedding = nn.Linear(max_z, n_atom_basis)

        self.store_neighbors = store_neighbors
        self.store_embeddings = store_embeddings
        self.n_edge_features = n_edge_features

        # if self.n_edge_features:
        #     self.edge_embedding = nn.Embedding(n_edge_features, self.n_interactions * 3 * n_atom_basis, padding_idx=0, max_norm=1.0)

        if type(activation) is str:
            if activation == 'swish':
                activation = F.silu
            elif activation == 'softplus':
                activation = snn.shifted_softplus

        self.filter_net = snn.Dense(
            n_rbf + n_edge_features, self.n_interactions * 3 * n_atom_basis, activation=None
        )

        self.interatomic_context_net = nn.ModuleList(
            [
                nn.Sequential(
                    snn.Dense(n_atom_basis, n_atom_basis, activation=activation),
                    snn.Dense(n_atom_basis, 3 * n_atom_basis, activation=None),
                )
                for _ in range(self.n_interactions)
            ]
        )

        self.intraatomic_context_net = nn.ModuleList(
            [
                nn.Sequential(
                    snn.Dense(
                        2 * n_atom_basis, n_atom_basis, activation=activation
                    ),
                    snn.Dense(n_atom_basis, 3 * n_atom_basis, activation=None),
                )
                for _ in range(self.n_interactions)
            ]
        )

        self.mu_channel_mix = nn.ModuleList(
            [
                nn.Sequential(
                    snn.Dense(n_atom_basis, 2 * n_atom_basis, activation=None, bias=False)
                )
                for _ in range(self.n_interactions)
            ]
        )

        # self.node_dec = nn.Sequential(snn.Dense(self.n_atom_basis, self.n_atom_basis, activation=F.silu),
        #                               snn.Dense(self.n_atom_basis, self.n_atom_basis))

        # self.graph_dec = nn.Sequential(snn.Dense(self.n_atom_basis, self.n_atom_basis, activation=F.silu),
        #                               snn.Dense(self.n_atom_basis, 1))    

    def forward(self, node_features, positions, neighbors, neighbor_mask, atom_mask):
        cell = None
        cell_offset = None
        # get interatomic vectors and distances
        rij, dir_ij = atom_distances(
            positions=positions,
            neighbors=neighbors,
            neighbor_mask=neighbor_mask,
            cell=cell,
            cell_offsets=cell_offset,
            return_vecs=True,
            normalize_vecs=True,
        )

        phi_ij = self.radial_basis(rij[..., None])

        fcut = self.cutoff_network(rij) * neighbor_mask
        # fcut = neighbor_mask
        fcut = fcut.unsqueeze(-1)

        filters = self.filter_net(phi_ij)

        # if self.n_edge_features:
        #     edge_types = inputs['edge_types']
        #     filters = filters + self.edge_embedding(edge_types)

        filters = filters * fcut
        filters = torch.split(filters, 3 * self.n_atom_basis, dim=-1)

        # initialize scalar and vector embeddings
        scalars = self.embedding(node_features)

        sshape = scalars.shape
        vectors = torch.zeros((sshape[0], sshape[1], 3, sshape[2]), device=scalars.device)

        for i in range(self.n_interactions):
            # message function
            h_i = self.interatomic_context_net[i](scalars)
            h_j, vectors_j = self.collect_neighbors(h_i, vectors, neighbors)

            # neighborhood context
            h_i = filters[i] * h_j

            dscalars, dvR, dvv = torch.split(h_i, self.n_atom_basis, dim=-1)
            dvectors = torch.einsum("bijf,bijd->bidf", dvR, dir_ij) + torch.einsum(
                "bijf,bijdf->bidf", dvv, vectors_j
            )
            dscalars = torch.sum(dscalars, dim=2)
            scalars = scalars + dscalars
            vectors = vectors + dvectors

            # update function
            mu_mix = self.mu_channel_mix[i](vectors)
            vectors_V, vectors_U = torch.split(mu_mix, self.n_atom_basis, dim=-1)
            mu_Vn = torch.norm(vectors_V, dim=2)

            ctx = torch.cat([scalars, mu_Vn], dim=-1)
            h_i = self.intraatomic_context_net[i](ctx)
            ds, dv, dsv = torch.split(h_i, self.n_atom_basis, dim=-1)
            dv = dv.unsqueeze(2) * vectors_U
            dsv = dsv * torch.einsum("bidf,bidf->bif", vectors_V, vectors_U)

            # calculate atomwise updates
            scalars = scalars + ds + dsv
            vectors = vectors + dv

        # h = self.node_dec(scalars)
        # h = h * atom_mask.unsqueeze(-1)
        # h = torch.sum(h, dim=1)
        # pred = self.graph_dec(h)
        # return pred.squeeze(1)
        return scalars

        # for layer in self.output_network:
        #     scalars, vectors = layer(scalars, vectors)
        # # include v in output to make sure all parameters have a gradient
        # pred = scalars + vectors.sum() * 0
        # pred = pred.squeeze(-1) * atom_mask
        # return torch.sum(pred, dim = -1)
        # # scalars = self.scalar_LN(scalars)
        # # vectors = self.vector_LN(vectors)

        

    def collect_neighbors(self, scalars, vectors, neighbors):
        nbh_size = neighbors.size()
        nbh = neighbors.view(-1, nbh_size[1] * nbh_size[2], 1)

        scalar_nbh = nbh.expand(-1, -1, scalars.size(2))
        scalars_j = torch.gather(scalars, 1, scalar_nbh)
        scalars_j = scalars_j.view(nbh_size[0], nbh_size[1], nbh_size[2], -1)

        vectors_nbh = nbh[..., None].expand(-1, -1, vectors.size(2), vectors.size(3))
        vectors_j = torch.gather(vectors, 1, vectors_nbh)
        vectors_j = vectors_j.view(nbh_size[0], nbh_size[1], nbh_size[2], 3, -1)
        return scalars_j, vectors_j

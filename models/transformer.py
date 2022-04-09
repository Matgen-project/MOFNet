import math, copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_model(d_atom, N=2, d_model=128, h=8, dropout=0.1, 
               lambda_attention=0.3, lambda_distance=0.3, trainable_lambda=False,
               N_dense=2, leaky_relu_slope=0.0, aggregation_type='mean', 
               dense_output_nonlinearity='relu', distance_matrix_kernel='softmax',
               use_edge_features=False, n_output=1,
               control_edges=False, integrated_distances=False, 
               scale_norm=False, n_generator_layers=1, d_feature=8, use_global_feature=False, d_mid_list=None, d_ff_list=None, use_ffn_only=False, adj_mask=None, adapter_finetune=False, **kwargs):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model, dropout, lambda_attention, lambda_distance, trainable_lambda, distance_matrix_kernel, use_edge_features, control_edges, integrated_distances, adj_mask)
    ff = PositionwiseFeedForward(d_model, N_dense, dropout, leaky_relu_slope, dense_output_nonlinearity)
    pooling = PoolingLayer(aggregation_type)
    if use_ffn_only:
        if d_ff_list is None and n_generator_layers > 1:
            d_ff_list = [d_model] * (n_generator_layers - 1)
        model = DNNGenerator(d_model, d_feature, n_output, leaky_relu_slope, dropout, scale_norm, d_ff_list)
    elif not use_global_feature:
        model = GraphTransformer(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout, scale_norm), N, scale_norm),
            Embeddings(d_model, d_atom, dropout),
            c(pooling),
            Generator(d_model, aggregation_type, n_output, n_generator_layers, leaky_relu_slope, dropout, scale_norm, d_ff_list))
    else:
        if d_ff_list is None and n_generator_layers > 1:
            d_ff_list = [d_model] * (n_generator_layers - 1)
        model = GraphTransformerWithGlobalFeature(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout, scale_norm), N, scale_norm),
            Embeddings(d_model, d_atom, dropout),
            c(pooling),
            GeneratorWithGlobalFeaturev3(d_model, d_feature, aggregation_type, n_output, dense_output_nonlinearity, leaky_relu_slope, dropout, scale_norm, d_mid_list, d_ff_list, adapter_finetune))
    for name,para in model.named_parameters():
        if name.endswith('weight'):
            if 'self_attn' in name:
                nn.init.xavier_uniform_(para,gain=1/math.sqrt(2))
            else:
                nn.init.xavier_uniform_(para)
        elif name.endswith('bias') and 'src_embed' not in name:
            nn.init.constant_(para,0.0)
        elif 'adapter_vec' in name:
            nn.init.constant_(para,0.0)

    return model


class GraphTransformer(nn.Module):
    def __init__(self, encoder, src_embed, pooling, generator):
        super(GraphTransformer, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.pooling = pooling
        self.generator = generator
        
    def forward(self, src, src_mask, adj_matrix, distances_matrix, global_feature):
        return self.predict(self.encode(src, src_mask, adj_matrix, distances_matrix, None))
    
    def encode(self, src, src_mask, adj_matrix, distances_matrix, edges_att):
        return self.pooling(self.encoder(self.src_embed(src), src_mask, adj_matrix, distances_matrix, edges_att),src_mask)
    
    def predict(self, out):
        return self.generator(out)

class GraphTransformerWithGlobalFeature(nn.Module):
    def __init__(self, encoder, src_embed, pooling, generator):
        super(GraphTransformerWithGlobalFeature, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.pooling = pooling
        self.generator = generator
        self.adapter_dim = None

    def set_adapter_dim(self,adapter_dim):
        self.adapter_dim = adapter_dim
        if hasattr(self.generator, 'adapter_dim'):
            self.generator.adapter_dim = adapter_dim
        
    def forward(self, src, src_mask, adj_matrix, distances_matrix, global_feature):
        return self.predict(self.encode(src, src_mask, adj_matrix, distances_matrix, None), global_feature)
    
    def encode(self, src, src_mask, adj_matrix, distances_matrix, edges_att):
        return self.pooling(self.encoder(self.src_embed(src), src_mask, adj_matrix, distances_matrix, edges_att),src_mask)
    
    def predict(self, out, global_feature):
        return self.generator(out, global_feature)
    
class Generator(nn.Module):
    
    def __init__(self, d_model, aggregation_type='mean', n_output=1, n_layers=1, 
                 leaky_relu_slope=0.01, dropout=0.0, scale_norm=False, d_ff_list=None):
        super(Generator, self).__init__()
        self.d_hidden = d_model
        if n_layers == 1:
            self.proj = nn.Linear(d_model, n_output)
        else:
            if d_ff_list is None:
                self.proj = nn.Linear(self.d_hidden, n_output)
            else:
                self.proj = []
                for d1,d2 in zip([self.d_hidden] + d_ff_list[:-1], d_ff_list):
                    self.proj.append(nn.Linear(d1, d2))
                    self.proj.append(nn.LeakyReLU(leaky_relu_slope))
                    self.proj.append(ScaleNorm(d2) if scale_norm else LayerNorm(d2))
                    self.proj.append(nn.Dropout(dropout))
                self.proj.append(nn.Linear(d_ff_list[-1], n_output))
                self.proj = torch.nn.Sequential(*self.proj)
        self.aggregation_type = aggregation_type

    def forward(self, x):
        projected = self.proj(x)
        return projected

class DNNGenerator(nn.Module):
    
    def __init__(self, d_model, d_feature, n_output=1,leaky_relu_slope=0.01,dropout=0.0,scale_norm=False, d_ff_list=None):
        super(DNNGenerator, self).__init__()
        self.d_feature = d_feature
        if d_ff_list is None:
            self.proj = nn.Linear(self.d_feature, n_output)
        else:
            self.proj = []
            for d1,d2 in zip([self.d_feature] + d_ff_list[:-1], d_ff_list):
                self.proj.append(nn.Linear(d1, d2))
                self.proj.append(nn.LeakyReLU(leaky_relu_slope))
                self.proj.append(ScaleNorm(d2) if scale_norm else LayerNorm(d2))
                self.proj.append(nn.Dropout(dropout))
            self.proj.append(nn.Linear(d_ff_list[-1], n_output))
            self.proj = torch.nn.Sequential(*self.proj)

    def forward(self, src, src_mask, adj_matrix, distances_matrix, global_feature):
        return self.proj(global_feature)

class BesselBasis(nn.Module):
    

    def __init__(self, cutoff=5.0, n_rbf=None):
        
        super(BesselBasis, self).__init__()
        
        freqs = torch.arange(1, n_rbf + 1) * math.pi / cutoff
        self.register_buffer("freqs", freqs)

    def forward(self, inputs):
        a = self.freqs[None, None, None, :]
        ax = inputs * a
        sinax = torch.sin(ax)

        norm = torch.where(inputs == 0, torch.tensor(1.0, device=inputs.device), inputs)
        y = sinax / norm

        return y

class CosineCutoff(nn.Module):
    
    def __init__(self, cutoff=5.0):
        super(CosineCutoff, self).__init__()
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))

    def forward(self, distances):
        
        cutoffs = 0.5 * (torch.cos(distances * np.pi / self.cutoff) + 1.0)
        
        cutoffs *= (distances < self.cutoff).float()
        return cutoffs

class PoolingLayer(nn.Module):
    def __init__(self, aggregation_type='mean'):
        super(PoolingLayer,self).__init__()
        self.aggregation_type = aggregation_type
    def forward(self, x, mask):
        mask = mask.unsqueeze(-1).float()
        out_masked = x * mask
        if self.aggregation_type == 'mean':
            out_sum = out_masked.sum(dim=1)
            mask_sum = mask.sum(dim=(1))
            out_avg_pooling = out_sum / mask_sum
        elif self.aggregation_type == 'sum':
            out_sum = out_masked.sum(dim=1)
            out_avg_pooling = out_sum
        elif self.aggregation_type == 'dummy_node':
            out_avg_pooling = out_masked[:,0]
        return out_avg_pooling

class GeneratorWithGlobalFeaturev3(nn.Module):
    
    def __init__(self, d_model, d_feature, aggregation_type='mean', n_output=1, dense_output_nonlinearity='relu',
                 leaky_relu_slope=0.01, dropout=0.0, scale_norm=False, d_mid_list=None, d_ff_list=None, adapter_finetune=False):
        super(GeneratorWithGlobalFeaturev3, self).__init__()
        c = copy.deepcopy
        if dense_output_nonlinearity == 'relu':
            self.act = nn.LeakyReLU(leaky_relu_slope)
        elif dense_output_nonlinearity == 'tanh':
            self.act = nn.Tanh()
        elif dense_output_nonlinearity == 'silu':
            self.act = nn.SiLU()

        if d_mid_list is not None:
            self.equal_proj = []
            for d1,d2 in zip([d_feature] + d_mid_list, d_mid_list + [d_model]):
                self.equal_proj.append(nn.Linear(d1,d2))
                self.equal_proj.append(c(self.act))
                self.equal_proj.append(ScaleNorm(d2) if scale_norm else LayerNorm(d2))
                self.equal_proj.append(nn.Dropout(dropout))
            self.equal_proj = torch.nn.Sequential(*self.equal_proj)
        else:
            self.equal_proj = nn.Sequential(nn.Linear(d_feature,d_model),nn.LeakyReLU(leaky_relu_slope),ScaleNorm(d_feature) if scale_norm else LayerNorm(d_feature),nn.Dropout(dropout))
        self.d_hidden = d_model * 2
        self.adapter_dim = None
        if d_ff_list is None:
            self.proj = nn.Linear(self.d_hidden, n_output)
        else:
            self.proj = []
            for d1,d2 in zip([self.d_hidden] + d_ff_list[:-1], d_ff_list):
                self.proj.append(nn.Linear(d1, d2))
                self.proj.append(c(self.act))
                self.proj.append(ScaleNorm(d2) if scale_norm else LayerNorm(d2))
                self.proj.append(nn.Dropout(dropout))
            self.proj.append(nn.Linear(d_ff_list[-1], n_output))
            self.proj = torch.nn.Sequential(*self.proj)
        self.aggregation_type = aggregation_type
        self.adapter_finetune = adapter_finetune
        if adapter_finetune:
            self.adapter_vec = torch.nn.Parameter(torch.Tensor(d_mid_list[0] if d_mid_list else d_model))

    def adapted_equal_proj(self,global_feature, adapter_dim):
        gf_ori = self.equal_proj[0](global_feature[...,:-adapter_dim]).repeat_interleave(adapter_dim,dim=0)
        gf_apt = global_feature[...,-adapter_dim:].reshape(-1,1) * self.adapter_vec
        return self.equal_proj[1:](gf_ori + gf_apt)

    def forward(self, x, global_feature):
        adapter_dim = self.adapter_dim
        out_avg_pooling = x
        if self.adapter_finetune:
            out_avg_pooling = out_avg_pooling.repeat_interleave(adapter_dim,dim=0)
            global_feature = self.adapted_equal_proj(global_feature, adapter_dim)
        else:
            global_feature = self.equal_proj(global_feature)
        out_avg_pooling = torch.cat([out_avg_pooling, global_feature],dim=1)
        # out_avg_pooling = global_feature
        projected = self.proj(out_avg_pooling)
        return projected    
    
class PositionGenerator(nn.Module):
    
    def __init__(self, d_model):
        super(PositionGenerator, self).__init__()
        self.norm = LayerNorm(d_model)
        self.proj = nn.Linear(d_model, 3)

    def forward(self, x, mask):
        mask = mask.unsqueeze(-1).float()
        out_masked = self.norm(x) * mask
        projected = self.proj(out_masked)
        return projected
    

def clones(module, N):
    
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    
    def __init__(self, layer, N, scale_norm):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = ScaleNorm(layer.size) if scale_norm else LayerNorm(layer.size)
        
    def forward(self, x, mask, adj_matrix, distances_matrix, edges_att):
        
        for layer in self.layers:
            x = layer(x, mask, adj_matrix, distances_matrix, edges_att)
        return self.norm(x)

    
class LayerNorm(nn.Module):
    
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
    
class ScaleNorm(nn.Module):

    def __init__(self, scale, eps=1e-5):
        super(ScaleNorm, self).__init__()
        self.scale = nn.Parameter(torch.tensor(math.sqrt(scale)))
        self.eps = eps
        
    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm

    
class SublayerConnection(nn.Module):
    
    def __init__(self, size, dropout, scale_norm):
        super(SublayerConnection, self).__init__()
        self.norm = ScaleNorm(size) if scale_norm else LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        
        return x + self.dropout(sublayer(self.norm(x)))

    
class EncoderLayer(nn.Module):
    
    def __init__(self, size, self_attn, feed_forward, dropout, scale_norm):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout, scale_norm), 2)
        self.size = size

    def forward(self, x, mask, adj_matrix, distances_matrix, edges_att):
        
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, adj_matrix, distances_matrix, edges_att, mask))
        return self.sublayer[1](x, self.feed_forward)

class FFNEncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout, scale_norm):
        super(FFNEncoderLayer, self).__init__()
        self.feed_forward = feed_forward
        self.sublayer = SublayerConnection(size, dropout, scale_norm)
        self.size = size

    def forward(self, x, mask, adj_matrix, distances_matrix, edges_att):
        return self.sublayer(x, self.feed_forward)    


class EdgeFeaturesLayer(nn.Module):
    def __init__(self, d_model, d_edge, h, dropout):
        super(EdgeFeaturesLayer, self).__init__()
        assert d_model % h == 0
        d_k = d_model // h
        self.linear = nn.Linear(d_edge, 1, bias=False)
        with torch.no_grad():
            self.linear.weight.fill_(0.25)

    def forward(self, x):
        p_edge = x.permute(0, 2, 3, 1)
        p_edge = self.linear(p_edge).permute(0, 3, 1, 2)
        return torch.relu(p_edge)
    

def attention(query, key, value, adj_matrix, distances_matrix, edges_att,
              mask=None, dropout=None, 
              lambdas=(0.3, 0.3, 0.4), trainable_lambda=False,
              distance_matrix_kernel=None, use_edge_features=False, control_edges=False,
              eps=1e-6, inf=1e12):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask.unsqueeze(1).repeat(1, query.shape[1], query.shape[2], 1) == 0, -inf)
    p_attn = F.softmax(scores, dim = -1)

    if use_edge_features:
        adj_matrix = edges_att.view(adj_matrix.shape)

    adj_matrix = adj_matrix / (adj_matrix.sum(dim=-1).unsqueeze(2) + eps)
    adj_matrix = adj_matrix.unsqueeze(1).repeat(1, query.shape[1], 1, 1)
    p_adj = adj_matrix
    
    p_dist = distances_matrix
    
    if trainable_lambda:
        softmax_attention, softmax_distance, softmax_adjacency = lambdas.cuda()
        p_weighted = softmax_attention * p_attn + softmax_distance * p_dist + softmax_adjacency * p_adj
    else:
        lambda_attention, lambda_distance, lambda_adjacency = lambdas
        p_weighted = lambda_attention * p_attn + lambda_distance * p_dist + lambda_adjacency * p_adj


    if dropout is not None:
        p_weighted = dropout(p_weighted)

    atoms_featrues = torch.matmul(p_weighted, value)     
    return atoms_featrues, p_weighted, p_attn

def cosineAttention(query, key, value, adj_matrix, distances_matrix, edges_att,
              mask=None, dropout=None, 
              lambdas=(0.3, 0.3, 0.4), trainable_lambda=False,
              distance_matrix_kernel=None, use_edge_features=False, control_edges=False,
              eps=1e-6, inf=1e12):
    
    q = query / (torch.norm(query, p = 2, dim = -1, keepdim = True).detach() + eps)
    k = key / (torch.norm(key, p = 2, dim = -1, keepdim = True).detach() + eps)

    scores = torch.matmul(q, k.transpose(-2, -1))
    if mask is not None:
        scores = scores.masked_fill(mask.unsqueeze(1).repeat(1, query.shape[1], query.shape[2], 1) == 0, 0)
    p_attn = F.softmax(scores, dim = -1)

    if use_edge_features:
        adj_matrix = edges_att.view(adj_matrix.shape)

    adj_matrix = adj_matrix / (adj_matrix.sum(dim=-1).unsqueeze(2) + eps)
    adj_matrix = adj_matrix.unsqueeze(1).repeat(1, query.shape[1], 1, 1)
    p_adj = adj_matrix
    
    p_dist = distances_matrix
    
    if trainable_lambda:
        softmax_attention, softmax_distance, softmax_adjacency = lambdas.cuda()
        p_weighted = softmax_attention * p_attn + softmax_distance * p_dist + softmax_adjacency * p_adj
    else:
        lambda_attention, lambda_distance, lambda_adjacency = lambdas
        p_weighted = lambda_attention * p_attn + lambda_distance * p_dist + lambda_adjacency * p_adj


    if dropout is not None:
        p_weighted = dropout(p_weighted)

    atoms_featrues = torch.matmul(p_weighted, value)     
    return atoms_featrues, p_weighted, p_attn

def attentionOnAdj(query, key, value, adj_matrix, distances_matrix, edges_att,
              mask=None, dropout=None, 
              lambdas=(0.3, 0.3, 0.4), trainable_lambda=False,
              distance_matrix_kernel=None, use_edge_features=False, control_edges=False,
              eps=1e-6, inf=1e12):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    scores = scores.masked_fill(adj_matrix.unsqueeze(1).repeat(1, query.shape[1], 1, 1) == 0, -inf)
    p_attn = F.softmax(scores, dim = -1)

    if use_edge_features:
        adj_matrix = edges_att.view(adj_matrix.shape)

    
    p_dist = distances_matrix
    
    if trainable_lambda:
        softmax_attention, softmax_distance, softmax_adjacency = lambdas.cuda()
        softmax_useful = softmax_attention + softmax_distance
        softmax_attention /= softmax_useful
        softmax_distance /= softmax_useful
        p_weighted = softmax_attention * p_attn + softmax_distance * p_dist
    else:
        lambda_attention, lambda_distance, lambda_adjacency = lambdas
        lambda_useful = lambda_attention + lambda_distance
        lambda_attention /= lambda_useful
        lambda_distance /= lambda_useful
        p_weighted = lambda_attention * p_attn + lambda_distance * p_dist


    if dropout is not None:
        p_weighted = dropout(p_weighted)

    atoms_featrues = torch.matmul(p_weighted, value)     
    return atoms_featrues, p_weighted, p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, lambda_attention=0.3, lambda_distance=0.3, trainable_lambda=False, 
                 distance_matrix_kernel='softmax', use_edge_features=False, control_edges=False, integrated_distances=False, adj_mask=False, n_rbf=20):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        
        self.d_k = d_model // h
        self.h = h
        self.trainable_lambda = trainable_lambda
        if trainable_lambda:
            lambda_adjacency = 1. - lambda_attention - lambda_distance
            lambdas_tensor = torch.tensor([lambda_attention, lambda_distance, lambda_adjacency], requires_grad=True)
            self.lambdas = torch.nn.Parameter(lambdas_tensor)
        else:
            lambda_adjacency = 1. - lambda_attention - lambda_distance
            self.lambdas = (lambda_attention, lambda_distance, lambda_adjacency)
            
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.use_filter = False
        if distance_matrix_kernel == 'softmax':
            self.distance_matrix_kernel = lambda x: F.softmax(-x, dim = -1)
        elif distance_matrix_kernel == 'exp':
            self.distance_matrix_kernel = lambda x: torch.exp(-x)
        elif distance_matrix_kernel == 'bessel':
            self.bessel = BesselBasis(n_rbf=n_rbf)
            self.cutoff = CosineCutoff()
            self.filter_act = nn.SiLU()
            self.distance_matrix_kernel = None
            self.filter_layer = nn.Linear(n_rbf, self.h)
            self.use_filter = True
        self.integrated_distances = integrated_distances
        self.use_edge_features = use_edge_features
        self.control_edges = control_edges
        self.adj_mask = adj_mask
        if use_edge_features:
            d_edge = 11 if not integrated_distances else 12
            self.edges_feature_layer = EdgeFeaturesLayer(d_model, d_edge, h, dropout)
        
    def forward(self, query, key, value, adj_matrix, distances_matrix, edges_att, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        
        if self.use_filter:
            distances_matrix = distances_matrix.unsqueeze(-1)
            distances_matrix_rbf = self.bessel(distances_matrix)
            p_dist = self.filter_layer(distances_matrix_rbf).masked_fill(mask.unsqueeze(-1).repeat(1, mask.shape[-1], 1, self.h)==0, 0)
            p_dist = self.filter_act(p_dist) * self.cutoff(distances_matrix)
            p_dist = p_dist.permute(0,3,1,2)
        else:
            distances_matrix = distances_matrix.masked_fill(mask.repeat(1, mask.shape[-1], 1) == 0, np.inf)
            distances_matrix = self.distance_matrix_kernel(distances_matrix)
            p_dist = distances_matrix.unsqueeze(1).repeat(1, query.shape[1], 1, 1)

        if self.use_edge_features:
            if self.integrated_distances:
                edges_att = torch.cat((edges_att, distances_matrix.unsqueeze(1)), dim=1)
            edges_att = self.edges_feature_layer(edges_att)
        
        if self.adj_mask is None:
            x, self.attn, self.self_attn = attention(query, key, value, adj_matrix, 
                                                    p_dist, edges_att,
                                                    mask=mask, dropout=self.dropout,
                                                    lambdas=self.lambdas,
                                                    trainable_lambda=self.trainable_lambda,
                                                    distance_matrix_kernel=self.distance_matrix_kernel,
                                                    use_edge_features=self.use_edge_features,
                                                    control_edges=self.control_edges)
        elif self.adj_mask == 'adj':
            x, self.attn, self.self_attn = attentionOnAdj(query, key, value, adj_matrix, 
                                                    p_dist, edges_att,
                                                    mask=mask, dropout=self.dropout,
                                                    lambdas=self.lambdas,
                                                    trainable_lambda=self.trainable_lambda,
                                                    distance_matrix_kernel=self.distance_matrix_kernel,
                                                    use_edge_features=self.use_edge_features,
                                                    control_edges=self.control_edges)
        elif self.adj_mask == 'qk':
            x, self.attn, self.self_attn = attention(query, query, value, adj_matrix, 
                                                    p_dist, edges_att,
                                                    mask=mask, dropout=self.dropout,
                                                    lambdas=self.lambdas,
                                                    trainable_lambda=self.trainable_lambda,
                                                    distance_matrix_kernel=self.distance_matrix_kernel,
                                                    use_edge_features=self.use_edge_features,
                                                    control_edges=self.control_edges)          
        elif self.adj_mask == 'cosine':
            x, self.attn, self.self_attn = cosineAttention(query, key, value, adj_matrix, 
                                                    p_dist, edges_att,
                                                    mask=mask, dropout=self.dropout,
                                                    lambdas=self.lambdas,
                                                    trainable_lambda=self.trainable_lambda,
                                                    distance_matrix_kernel=self.distance_matrix_kernel,
                                                    use_edge_features=self.use_edge_features,
                                                    control_edges=self.control_edges)            
        elif self.adj_mask == 'cosineqk':
            x, self.attn, self.self_attn = cosineAttention(query, query, value, adj_matrix, 
                                                    p_dist, edges_att,
                                                    mask=mask, dropout=self.dropout,
                                                    lambdas=self.lambdas,
                                                    trainable_lambda=self.trainable_lambda,
                                                    distance_matrix_kernel=self.distance_matrix_kernel,
                                                    use_edge_features=self.use_edge_features,
                                                    control_edges=self.control_edges)  

        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, N_dense, dropout=0.1, leaky_relu_slope=0.0, dense_output_nonlinearity='relu'):
        super(PositionwiseFeedForward, self).__init__()
        self.N_dense = N_dense
        self.linears = clones(nn.Linear(d_model, d_model), N_dense)
        self.dropout = clones(nn.Dropout(dropout), N_dense)
        self.leaky_relu_slope = leaky_relu_slope
        if dense_output_nonlinearity == 'relu':
            self.dense_output_nonlinearity = lambda x: F.leaky_relu(x, negative_slope=self.leaky_relu_slope)
        elif dense_output_nonlinearity == 'tanh':
            self.tanh = torch.nn.Tanh()
            self.dense_output_nonlinearity = lambda x: self.tanh(x)
        elif dense_output_nonlinearity == 'silu':
            self.silu = nn.SiLU()
            self.dense_output_nonlinearity = lambda x: self.silu(x)
        elif dense_output_nonlinearity == 'none':
            self.dense_output_nonlinearity = lambda x: x
            

    def forward(self, x):
        if self.N_dense == 0:
            return x
        
        for i in range(len(self.linears)-1):
            x = self.dropout[i](F.leaky_relu(self.linears[i](x), negative_slope=self.leaky_relu_slope))
            
        return self.dropout[-1](self.dense_output_nonlinearity(self.linears[-1](x)))

class Embeddings(nn.Module):
    def __init__(self, d_model, d_atom, dropout):
        super(Embeddings, self).__init__()
        self.lut = nn.Linear(d_atom, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.lut(x))

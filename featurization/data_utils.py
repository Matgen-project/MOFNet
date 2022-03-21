"""

"""

import logging
import os
import pickle

import numpy as np
import pandas as pd
import torch
#torch.multiprocessing.set_start_method('spawn')
#from rdkit import Chem
#from rdkit.Chem import AllChem
#from rdkit.Chem import MolFromSmiles
from sklearn.metrics import pairwise_distances
from torch.utils.data import Dataset, dataset
import json
import copy

#use_cuda = torch.cuda.is_available()
use_cuda = False
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
DoubleTensor = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor

def cutname(ori_name):
    ori_name = ori_name[:-3]
    if ori_name.endswith('out.'):
        ori_name = ori_name[:-4]
    elif ori_name.endswith('faps.'):
        ori_name = ori_name[:-5]
    return ori_name + 'p'


def load_hmof_data(dataset_path):
    global_features_label = pd.read_csv(dataset_path + '/global_features/hmof/global_feature_label.csv')
    data_x = global_features_label.iloc[:, 0].values
    data_f = global_features_label.iloc[:,1:].values.astype(np.float32)
    mean = np.mean(data_f,axis=0)
    std = np.std(data_f,axis=0)
    #print(mean,std)
    data_f = (data_f - mean) / std
    data_dict = {}
    for i in range(data_x.shape[0]):
        data_dict[cutname(data_x[i])] = data_f[i]
    return data_dict, mean, std

def load_hmof_list(listname, data_dict):
    with open(listname, 'r') as f:
        l = f.readlines()
    filtered_l = []
    for fn in l:
        fn = fn.strip()
        if fn in data_dict:
            filtered_l.append(fn)
    return filtered_l

# def load_data_from_df(dataset_path, add_dummy_node=True, use_global_features=False):
#     """Load and featurize data stored in a CSV file.

#     Args:
#         dataset_path (str): A path to the CSV file containing the data. It should have two columns:
#                             the first one contains SMILES strings of the compounds,
#                             the second one contains labels.
#         add_dummy_node (bool): If True, a dummy node will be added to the molecular graph. Defaults to True.
#         one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded. Defaults to False.

#     Returns:
#         A tuple (X, y) in which X is a list of graph descriptors (node features, adjacency matrices, distance matrices),
#         and y is a list of the corresponding labels.
#     """

#     feature_path = dataset_path + 'ads_co2_1e5.csv'
#     data_df = pd.read_csv(feature_path,header=None)

#     data_x = data_df.iloc[:, 0].values
#     data_y = data_df.iloc[:, 1].values

#     if data_y.dtype == np.float64:
#         data_y = data_y.astype(np.float32)

#     x_all, y_all = load_data_from_processed(dataset_path, data_x, data_y, add_dummy_node=add_dummy_node)

#     if use_global_features:
#         f_all = load_data_with_global_features(dataset_path, data_x)
#         return x_all, f_all, y_all

#     return x_all, y_all

def load_data_from_df(dataset_path, gas_type, pressure, add_dummy_node=True, use_global_features=False, return_names=False):
    """Load and featurize data stored in a CSV file.

    Args:
        dataset_path (str): A path to the CSV file containing the data. It should have two columns:
                            the first one contains SMILES strings of the compounds,
                            the second one contains labels.
        add_dummy_node (bool): If True, a dummy node will be added to the molecular graph. Defaults to True.
        one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded. Defaults to False.

    Returns:
        A tuple (X, y) in which X is a list of graph descriptors (node features, adjacency matrices, distance matrices),
        and y is a list of the corresponding labels.
    """
    print(dataset_path + f'/label/{gas_type}/{gas_type}_ads_all.csv')
    data_df = pd.read_csv(dataset_path + f'/label/{gas_type}/{gas_type}_ads_all.csv',header=0)
    
    data_x = data_df['name'].values
    #print(pressure)
    #print(data_df[pressure])
    if pressure == 'all':
        data_y = data_df.iloc[:,1:].values
    else:
        data_y = data_df[pressure].values

    if data_y.dtype == np.float64:
        data_y = data_y.astype(np.float32)

    x_all, y_all, name_all = load_data_from_processed(dataset_path, data_x, data_y, add_dummy_node=add_dummy_node)

    if return_names:
        x_all = (x_all, name_all)

    if use_global_features:
        f_all = load_data_with_global_features(dataset_path, name_all, gas_type)
        if pressure == 'all':
            return x_all, f_all, y_all, data_df.columns.values[1:]
        return x_all, f_all, y_all

    if pressure == 'all':
        return x_all, y_all, data_df.columns.values[1:]
    return x_all, y_all

def norm_str(ori):
    ori = ori.split('.')[0].split('-')
    if ori[-1] == 'clean':
        ori = ori[:-1]
    elif ori[-2] == 'clean':
        ori = ori[:-2]
    return '-'.join(ori[1:])

def load_predict_data(dataset_path):
    files = os.listdir(dataset_path)
    csvs = [_ for _ in files if _.endswith('.csv')]
    data_x, data_y = None, None
    for f in csvs:
        data_df = pd.read_csv(os.path.join(dataset_path, f), header=0)
        data_x = np.concatenate([data_x, data_df['name'].values]) if data_x is not None else data_df['name'].values
        data_y = np.concatenate([data_y, data_df.iloc[:,1:].values]) if data_y is not None else data_df.iloc[:,1:].values
    x_all, f_all, n_all = [], [], []
    for x,y in zip(data_x, data_y):
        with open(os.path.join(dataset_path,x+'.p'),'rb') as p:
            afm, adj, dist = pickle.load(p)
        afm, adj, dist = add_dummy_node_func(afm, adj, dist)
        x_all.append([afm, adj, dist])
        f_all.append(y)
        n_all.append(x)
    return x_all, f_all, n_all

def load_real_data(dataset_path, gas_type, add=None):
    gas_type_dict = {"CH4":"Methane", "CO2":"Carbon Dioxide", "N2":"Nitrogen"}
    bound = {"CH4":5e4,"CO2":1e4,"N2":2e2}
    b = bound[gas_type]
    # data_df = pd.read_csv(dataset_path + f'/global_feat/exp_geo_all.csv', header=0)
    data_df = pd.read_csv(dataset_path + f'/global_feat/{add}.csv', header=0)
    data_x = data_df['name'].values
    data_y = data_df.iloc[:,2:].values
    global_dic = {}
    for x,y in zip(data_x, data_y):
        global_dic[x] = y
    with open(dataset_path + '/label_nist/all.json') as f:
        labels = json.load(f)[gas_type_dict[gas_type]]['data']
    label_dict = {_['name']:_["isotherm_data"] for _ in labels}
    # label_dict = {}
    # for label in labels:
    #     if label['name'] not in label_dict:
    #         label_dict[label['name']] = []
    #     label_dict[label['name']] += label['isotherm_data']
    
    with open(dataset_path + f'/{gas_type_dict[gas_type]}_list','r') as f:
        ls = f.readlines()
    ls = [_.strip() for _ in ls]
    X_all, y_all, f_all, p_all, n_all = [],[],[],[],[]
    for l in ls:
        if l[:-2] not in global_dic:
            continue
        gf = global_dic[l[:-2]]
        afm, adj, dist = pickle.load(open(dataset_path + f'/local_feat/{l}', "rb"))
        afm, adj, dist = add_dummy_node_func(afm, adj, dist)
        iso = label_dict[norm_str(l)]
        p,y = [],[]
        for _ in iso:
            if _['pressure'] > 0:
                p.append(_['pressure'])
                y.append(_['adsorption'])
        if len(p) == 0:
            continue
        X_all.append([afm,adj,dist])
        f_all.append(gf)
        p_all.append(p)
        y_all.append(y)
        n_all.append(norm_str(l))
    return X_all, f_all, y_all, p_all, n_all



def load_data_with_global_features(dataset_path, processed_files, gas_type):
    global_feature_path = dataset_path + f'/label/{gas_type}/{gas_type}_global_features_update.csv'
    #print(global_feature_path)
    data_df = pd.read_csv(global_feature_path,header=0)
    data_x = data_df.iloc[:, 0].values
    data_f = data_df.iloc[:,1:].values.astype(np.float32)
    data_dict = {}
    for i in range(data_x.shape[0]):
        data_dict[data_x[i]] = data_f[i]
    f_all = [data_dict[_] for _ in processed_files]
    return f_all



def load_data_from_processed(dataset_path, processed_files, labels, add_dummy_node=True):
    """Load and featurize data from lists of SMILES strings and labels.

    Args:
        data_set_path (list(str)): data path for processed files.
        processed_files (list(str)): processed files name.
        labels (list[float]): A list of the corresponding labels.
        add_dummy_node (bool): If True, a dummy node will be added to the molecular graph. Defaults to True.

    Returns:
        A tuple (X, y) in which X is a list of graph descriptors (node features, adjacency matrices, distance matrices),
        and y is a list of the corresponding labels.
    """
    x_all, y_all, name_all = [], [], []

    for files, label in zip(processed_files, labels):
        
        data_file = dataset_path + '/processed_unit/' + files + '.p'
        #print(data_file)
        try:
            afm, adj, dist = pickle.load(open(data_file, "rb"))
            if add_dummy_node:
                afm, adj, dist = add_dummy_node_func(afm, adj, dist)
            #print(afm,adj,dist)
            x_all.append([afm, adj, dist])
            y_all.append([label])
            name_all.append(files)
        except:
            pass

    return x_all, y_all, name_all

def add_dummy_node_func(node_features, adj_matrix, dist_matrix):
    m = np.zeros((node_features.shape[0] + 1, node_features.shape[1] + 1))
    m[1:, 1:] = node_features
    m[0, 0] = 1.
    node_features = m

    m = np.ones((adj_matrix.shape[0] + 1, adj_matrix.shape[1] + 1))
    m[1:, 1:] = adj_matrix
    adj_matrix = m

    m = np.full((dist_matrix.shape[0] + 1, dist_matrix.shape[1] + 1), 1e6)
    m[1:, 1:] = dist_matrix
    # row, col = np.diag_indices_from(m)
    # m[row,col] = 1e6
    dist_matrix = m

    return node_features, adj_matrix, dist_matrix


class Molecule:
    """
        Class that represents a train/validation/test datum
        - self.label: 0 neg, 1 pos -1 missing for different target.
    """

    def __init__(self, x, y, index, feature = None):
        self.node_features = x[0]
        self.adjacency_matrix = x[1]
        self.distance_matrix = x[2]
        self.y = y
        self.index = index
        self.global_feature = feature


class MolDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """

    def __init__(self, data_list):
        """
        @param data_list: list of Molecule objects
        """
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, key):
        if type(key) == slice:
            return MolDataset(self.data_list[key])
        return self.data_list[key]


class RealMOFDataset(Dataset):
    def __init__(self, data_list, pressure_list, ori_point):
        self.data_list = data_list
        self.pressure_list = pressure_list
        self.ori_point = np.log(np.float32(ori_point))
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self,key):
        if type(key) == slice:
            return RealMOFDataset(self.data_list[key], self.pressure_list[key], self.ori_point)
        tar_mol = copy.deepcopy(self.data_list[key])
        tar_p = np.log(self.pressure_list[key]) - self.ori_point
        tar_mol.global_feature = np.append(tar_mol.global_feature, tar_p)
        tar_mol.y = tar_mol.y
        return tar_mol

class MolDatasetPressureVer(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """

    def __init__(self, data_list, pressure_list, mask_point=None, is_train=True, tar_point=None):
        """
        @param data_list: list of Molecule objects
        """
        self.data_list = data_list
        self.pressure_list = pressure_list
        self.mask_point = mask_point
        self.is_train = is_train
        self.tar_point = tar_point
        if is_train:
            self.use_idx = np.where(pressure_list != mask_point)[0]
        else:
            self.use_idx = np.where(pressure_list == tar_point)[0]
        self.calcMid()

    def __len__(self):
        return len(self.data_list)

    def toStr(self):
        return {"data_list":self.data_list,"pressure_list":self.pressure_list,"mask_point":self.mask_point,"is_train":self.is_train, "tar_point":self.tar_point}
    def __getitem__(self, key):
        if type(key) == slice:
            return MolDataset(self.data_list[key], self.pressure_list, self.mask_point, self.is_train)
        tar_mol = copy.deepcopy(self.data_list[key])
        # tar_idx = np.random.choice(self.use_idx)
        # tar_p = np.log(float(self.pressure_list[tar_idx]))
        # tar_p = self.float_pressure[tar_idx] - self.mid
        # tar_mol.global_feature = np.append(tar_mol.global_feature, tar_p)
        # tar_mol.y = [tar_mol.y[0][tar_idx]]
        # return tar_mol
        if self.is_train:
            tar_p = self.float_pressure - self.mid
            tar_mol.global_feature = np.append(tar_mol.global_feature, tar_p)
            tar_mol.y = tar_mol.y[0]
        else:
            tar_idx = self.use_idx
            tar_p = self.float_pressure[tar_idx] - self.mid
            tar_mol.global_feature = np.append(tar_mol.global_feature, tar_p)
            tar_mol.y = [tar_mol.y[0][tar_idx]]
        return tar_mol

    def changeTarPoint(self,tar_point):
        self.tar_point = tar_point
        if not tar_point:
            self.is_train = True
        else:
            self.is_train = False
        if not self.is_train:
            self.use_idx = np.where(self.pressure_list == tar_point)[0]

    def calcMid(self):
        self.float_pressure = np.log(self.pressure_list.astype(np.float))
        self.mid = np.log(np.float(self.mask_point))

class MOFDataset(Dataset):
    def __init__(self, mof_list, mof_dir, feature_dict):
        super(MOFDataset,self).__init__()
        self.mof_list = mof_list
        self.mof_dir = mof_dir
        self.feature_dict = feature_dict
    def __len__(self) -> int:
        return len(self.mof_list)
    def __getitem__(self, key):
        if isinstance(key, slice):
            return MOFDataset(self.mof_list[key], self.mof_dir, self.feature_dict)
        mol_name = self.mof_list[key]
        with open(self.mof_dir + '/' + mol_name, 'rb') as f:
            afm, adj, dist = pickle.load(f)
        afm, adj, dist = add_dummy_node_func(afm, adj, dist)
        gf_l = self.feature_dict[mol_name]
        #print(mol_name,gf_l)
        return afm, adj, dist, gf_l[:-1], gf_l[-1]


def pad_array(array, shape, dtype=np.float32):
    """Pad a 2-dimensional array with zeros.

    Args:
        array (ndarray): A 2-dimensional array to be padded.
        shape (tuple[int]): The desired shape of the padded array.
        dtype (data-type): The desired data-type for the array.

    Returns:
        A 2-dimensional array of the given shape padded with zeros.
    """
    padded_array = np.zeros(shape, dtype=dtype)
    padded_array[:array.shape[0], :array.shape[1]] = array
    return padded_array


def mol_collate_func(batch):
    """Create a padded batch of molecule features.

    Args:
        batch (list[Molecule]): A batch of raw molecules.

    Returns:
        A list of FloatTensors with padded molecule features:
        adjacency matrices, node features, distance matrices, and labels.
    """
    adjacency_list, distance_list, features_list = [], [], []
    labels = []

    max_size = 0
    for molecule in batch:
        if type(molecule.y[0]) == np.ndarray:
            labels.append(molecule.y[0])
        else:
            labels.append(molecule.y)
        if molecule.adjacency_matrix.shape[0] > max_size:
            max_size = molecule.adjacency_matrix.shape[0]

    for molecule in batch:
        adjacency_list.append(pad_array(molecule.adjacency_matrix, (max_size, max_size)))
        distance_list.append(pad_array(molecule.distance_matrix, (max_size, max_size)))
        features_list.append(pad_array(molecule.node_features, (max_size, molecule.node_features.shape[1])))

    return [FloatTensor(features) for features in (adjacency_list, features_list, distance_list, labels)]

def mol_collate_func_gf(batch):
    """Create a padded batch of molecule features with additional global features.

    Args:
        batch (list[Molecule]): A batch of raw molecules.

    Returns:
        A list of FloatTensors with padded molecule features:
        adjacency matrices, node features, distance matrices, global features and labels.
    """
    adjacency_list, distance_list, features_list, global_features_list = [], [], [], []
    labels = []

    max_size = 0
    for molecule in batch:
        if type(molecule.y[0]) == np.ndarray:
            labels.append(molecule.y[0])
        else:
            labels.append(molecule.y)
        if molecule.adjacency_matrix.shape[0] > max_size:
            max_size = molecule.adjacency_matrix.shape[0]

    for molecule in batch:
        adjacency_list.append(pad_array(molecule.adjacency_matrix, (max_size, max_size)))
        distance_list.append(pad_array(molecule.distance_matrix, (max_size, max_size)))
        features_list.append(pad_array(molecule.node_features, (max_size, molecule.node_features.shape[1])))
        global_features_list.append(molecule.global_feature)

    return [FloatTensor(features) for features in (adjacency_list, features_list, distance_list, global_features_list, labels)]  

def mof_collate_func_gf(batch):
    """Create a padded batch of molecule features with additional global features.

    Args:
        batch (list[Molecule]): A batch of MOFs.

    Returns:
        A list of FloatTensors with padded molecule features:
        adjacency matrices, node features, distance matrices, global features and labels.
    """
    adjacency_list, distance_list, features_list, global_features_list = [], [], [], []
    labels = []

    max_size = 0
    for afm,adj,dist,gf,lb in batch:
        labels.append([lb])
        if adj.shape[0] > max_size:
            max_size = adj.shape[0]

    for molecule in batch:
        adjacency_list.append(pad_array(adj, (max_size, max_size)))
        distance_list.append(pad_array(dist, (max_size, max_size)))
        features_list.append(pad_array(afm, (max_size, afm.shape[1])))
        global_features_list.append(gf)

    return [FloatTensor(features) for features in (adjacency_list, features_list, distance_list, global_features_list, labels)]    

def construct_dataset(x_all, y_all):
    """Construct a MolDataset object from the provided data.

    Args:
        x_all (list): A list of molecule features.
        y_all (list): A list of the corresponding labels.

    Returns:
        A MolDataset object filled with the provided data.
    """
    output = [Molecule(data[0], data[1], i)
              for i, data in enumerate(zip(x_all, y_all))]
    return MolDataset(output)

def construct_dataset_gf(x_all, f_all, y_all):
    """Construct a MolDataset object from the provided data.

    Args:
        x_all (list): A list of atom features of molecules.
        f_all (list): A list of molecule global feature.
        y_all (list): A list of the corresponding labels.

    Returns:
        A MolDataset object filled with the provided data.
    """
    output = [Molecule(data[0], data[2], i, data[1])
              for i, data in enumerate(zip(x_all, f_all, y_all))]
    return MolDataset(output)

def construct_dataset_gf_pressurever(x_all, f_all, y_all, pressure_list, is_train=True, mask_point=None, tar_point=None):
    """Construct a MolDataset object from the provided data.

    Args:
        x_all (list): A list of atom features of molecules.
        f_all (list): A list of molecule global feature.
        y_all (list): A list of the corresponding labels.

    Returns:
        A MolDataset object filled with the provided data.
    """
    output = [Molecule(data[0], data[2], i, data[1])
              for i, data in enumerate(zip(x_all, f_all, y_all))]
    return MolDatasetPressureVer(output, pressure_list, is_train=is_train, mask_point=mask_point,tar_point=tar_point)

def construct_dataset_real(x_all, f_all, y_all, pressure_list, tar_point=None):
    """Construct a MolDataset object from the provided data.

    Args:
        x_all (list): A list of atom features of molecules.
        f_all (list): A list of molecule global feature.
        y_all (list): A list of the corresponding labels.

    Returns:
        A MolDataset object filled with the provided data.
    """
    output = [Molecule(data[0], data[2], i, data[1])
              for i, data in enumerate(zip(x_all, f_all, y_all))]
    return RealMOFDataset(output, pressure_list, ori_point=tar_point)


def construct_loader(x, y, batch_size, shuffle=True):
    """Construct a data loader for the provided data.

    Args:
        x (list): A list of molecule features.
        y (list): A list of the corresponding labels.
        batch_size (int): The batch size.
        shuffle (bool): If True the data will be loaded in a random order. Defaults to True.

    Returns:
        A DataLoader object that yields batches of padded molecule features.
    """
    data_set = construct_dataset(x, y)
    loader = torch.utils.data.DataLoader(dataset=data_set,
                                         batch_size=batch_size,
					 num_workers=8,
                                         collate_fn=mol_collate_func,
					 pin_memory=True,
                                         shuffle=shuffle)
    return loader

def construct_loader_gf(x,f,y, batch_size, shuffle=True):
    """Construct a data loader for the provided data.

    Args:
        x (list): A list of atom features of molecules.
        f (list): A list of molecule global features.
        y (list): A list of the corresponding labels.
        batch_size (int): The batch size.
        shuffle (bool): If True the data will be loaded in a random order. Defaults to True.

    Returns:
        A DataLoader object that yields batches of padded molecule features.
    """
    data_set = construct_dataset_gf(x, f, y)
    loader = torch.utils.data.DataLoader(dataset=data_set,
                                         batch_size=batch_size,
					 num_workers=8,
                                         collate_fn=mol_collate_func_gf,
					 pin_memory=True,
                                         shuffle=shuffle)
    return loader

def construct_loader_gf_pressurever(data_set, batch_size, shuffle=True):
    """Construct a data loader for the provided data.

    Args:
        x (list): A list of atom features of molecules.
        f (list): A list of molecule global features.
        y (list): A list of the corresponding labels.
        batch_size (int): The batch size.
        shuffle (bool): If True the data will be loaded in a random order. Defaults to True.

    Returns:
        A DataLoader object that yields batches of padded molecule features.
    """
    # data_set = construct_dataset_gf(x, f, y, pressure_list, is_train=is_train, mask_point=mask_point,tar_point=tar_point)
    loader = torch.utils.data.DataLoader(dataset=data_set,
                                         batch_size=batch_size,
					 num_workers=0,
                                         collate_fn=mol_collate_func_gf,
					 pin_memory=False,
                                         shuffle=shuffle)
    return loader

def construct_loader_mof(mof_list, mof_dir, feature_dict, batch_size, shuffle=True):
    """Construct a data loader for the provided data.

    Args:
        mof_list (list): A list of MOF filenames.
        mof_dir (str): A path of directory containing MOF files.
        feature_dict (dict): A dict of global feature and labels.
        batch_size (int): The batch size.
        shuffle (bool): If True the data will be loaded in a random order. Defaults to True.

    Returns:
        A DataLoader object that yields batches of padded molecule features.
    """
    data_set = MOFDataset(mof_list, mof_dir, feature_dict)
    loader = torch.utils.data.DataLoader(dataset=data_set,
                                         batch_size=batch_size,
					 num_workers=8,
                                         collate_fn=mof_collate_func_gf,
					 pin_memory=True,
                                         shuffle=shuffle)
    return loader

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_data = None
            return
        with torch.cuda.stream(self.stream):
            self.next_data = tuple(_.cuda(non_blocking=True) for _ in self.next_data)
    
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_data
        self.preload()
        return batch

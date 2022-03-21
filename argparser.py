from argparse import ArgumentParser, Namespace
import os

def parse_train_args():
    """
    Parses arguments for training (includes modifying/validating arguments).

    :return: A Namespace containing the parsed, modified, and validated args.
    """
    parser = ArgumentParser()
    add_data_args(parser)
    add_train_args(parser)
    args = parser.parse_args()
    args = vars(args)
    lambda_mat = [float(_) for _ in args['weight_split'].split(',')]
    assert len(lambda_mat) == 3
    lambda_sum = sum(lambda_mat)
    args['lambda_attention'] = lambda_mat[0] / lambda_sum
    args['lambda_distance'] = lambda_mat[-1] / lambda_sum
    if args['d_mid_list'] == 'None':
        args['d_mid_list'] = []
    else:
        args['d_mid_list'] = [int(_) for _ in args['d_mid_list'].split(',')]
    makedirs(args['save_dir'] + f"/{args['gas_type']}_{args['pressure']}/")
    return args

def parse_predict_args():
    """
    Parses arguments for training (includes modifying/validating arguments).

    :return: A Namespace containing the parsed, modified, and validated args.
    """
    parser = ArgumentParser()
    add_data_args(parser)
    add_train_args(parser)
    args = parser.parse_args()
    args = vars(args)
    lambda_mat = [float(_) for _ in args['weight_split'].split(',')]
    assert len(lambda_mat) == 3
    lambda_sum = sum(lambda_mat)
    args['lambda_attention'] = lambda_mat[0] / lambda_sum
    args['lambda_distance'] = lambda_mat[-1] / lambda_sum
    if args['d_mid_list'] == 'None':
        args['d_mid_list'] = []
    else:
        args['d_mid_list'] = [int(_) for _ in args['d_mid_list'].split(',')]
    p_cond = args['pressure'].split(',')
    assert len(p_cond) == 3
    args['pressure'] = (float(p_cond[0]), float(p_cond[1]), int(p_cond[2]))
    return args

def parse_baseline_args():
    parser = ArgumentParser()
    add_data_args(parser)
    add_baseline_args(parser)
    args = parser.parse_args()
    args = vars(args)
    makedirs(args['save_dir'] + f"/{args['gas_type']}_{args['pressure']}/")
    return args    

def parse_finetune_args():
    parser = ArgumentParser()
    add_data_args(parser)
    add_finetune_args(parser)
    args = parser.parse_args()
    args = vars(args)
    makedirs(args['save_dir'] + f"/{args['gas_type']}_{args['pressure']}/")
    return args

def parse_ml_args():
    """
    Parses arguments for training (includes modifying/validating arguments).

    :return: A Namespace containing the parsed, modified, and validated args.
    """
    parser = ArgumentParser()
    add_data_args(parser)
    add_ml_args(parser)
    args = parser.parse_args()
    args = vars(args)
    makedirs(args['save_dir'] + f"/{args['ml_type']}/{args['gas_type']}_{args['pressure']}/")
    return args    

def parse_hmof_args():
    """
    Parses arguments for training (includes modifying/validating arguments).

    :return: A Namespace containing the parsed, modified, and validated args.
    """
    parser = ArgumentParser()
    add_data_args(parser, data_type='hMOF')
    add_train_args(parser)
    args = parser.parse_args()
    args = vars(args)
    lambda_mat = [float(_) for _ in args['weight_split'].split(',')]
    assert len(lambda_mat) == 3
    lambda_sum = sum(lambda_mat)
    args['lambda_attention'] = lambda_mat[0] / lambda_sum
    args['lambda_distance'] = lambda_mat[-1] / lambda_sum
    if args['d_mid_list'] == 'None':
        args['d_mid_list'] = []
    else:
        args['d_mid_list'] = [int(_) for _ in args['d_mid_list'].split(',')]
    makedirs(args['save_dir'])
    return args

def makedirs(path: str, isfile: bool = False):
    """
    Creates a directory given a path to either a directory or file.

    If a directory is provided, creates that directory. If a file is provided (i.e. isfiled == True),
    creates the parent directory for that file.

    :param path: Path to a directory or file.
    :param isfile: Whether the provided path is a directory or file.
    """
    if isfile:
        path = os.path.dirname(path)
    if path != '':
        os.makedirs(path, exist_ok=True)

def add_ml_args(parser: ArgumentParser):
    """
    Adds data arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """
    parser.add_argument('--ml_type', type=str, default='RF',
                    help='ML algorithm, SVR/DT/RF.')

    parser.add_argument('--seed', type=int, default=9999,
                    help='Random seed to use when splitting data into train/val/test sets.'
                            'When `num_folds` > 1, the first fold uses this seed and all'
                            'subsequent folds add 1 to the seed.') 
    parser.add_argument('--fold', type=int, default=10,
                    help='Fold num.') 

def add_data_args(parser: ArgumentParser, data_type='CSD'):
    """
    Adds data arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """
    parser.add_argument('--data_dir', type=str,
                    help='Dataset directory, containing label/ and processed/ subdirectories.')

    parser.add_argument('--save_dir', type=str, 
                    help='Model directory.')        
    
    if data_type == 'CSD':

        parser.add_argument('--gas_type', type=str,
                        help='Gas type for prediction.')

        parser.add_argument('--pressure', type=str, 
                        help='Pressure condition for prediction.')

        parser.add_argument('--name', type=str, default='',
                        help='Target MOF name.')

def add_finetune_args(parser: ArgumentParser):
    """
    Adds finetune arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """
    parser.add_argument('--ori_dir', type=str,
                    help='Pretrained model directory, containing model of different Folds.')
    
    parser.add_argument('--epoch', type=int, default=100,
                    help='Epoch num.')  

    parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size.')   

    parser.add_argument('--fold', type=int, default=10,
                    help='Fold num.') 

    parser.add_argument('--lr', type=float, default=0.0007,
                    help='Learning rate.')

    parser.add_argument('--adapter_dim', type=int, default=8,
                    help='Adapted vector dimension')

    parser.add_argument('--seed', type=int, default=9999,
                    help='Random seed to use when splitting data into train/val/test sets.')

def add_baseline_args(parser: ArgumentParser):
    """
    Adds training arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """
    parser.add_argument('--model_name',type=str,default='gin',
                help='Baseline Model, gin/egnn/schnet/painn.')

    parser.add_argument('--gpu', type=int,
                    help='GPU id to allocate.')

    parser.add_argument('--seed', type=int, default=9999,
                    help='Random seed to use when splitting data into train/val/test sets.')

    parser.add_argument('--d_model', type=int, default=1024,
                    help='Hidden size of baseline model.')

    parser.add_argument('--N', type=int, default=2,
                    help='Layer num of baseline model.')

    parser.add_argument('--use_global_feature', action='store_true',
                help='Whether to use global features(graph-level features).')

    parser.add_argument('--warmup_step', type=int, default=2000,
                    help='Warmup steps.')

    parser.add_argument('--epoch', type=int, default=100,
                    help='Epoch num.')  

    parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size.')   

    parser.add_argument('--fold', type=int, default=10,
                    help='Fold num.')    

    parser.add_argument('--lr', type=float, default=0.0007,
                    help='Maximum learning rate, (warmup_step * d_model) ** -0.5 .')

def add_train_args(parser: ArgumentParser):
    """
    Adds training arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """
    parser.add_argument('--seed', type=int, default=9999,
                    help='Random seed to use when splitting data into train/val/test sets.')

    parser.add_argument('--d_model', type=int, default=1024,
                    help='Hidden size of transformer model.')

    parser.add_argument('--N', type=int, default=2,
                    help='Layer num of transformer model.')

    parser.add_argument('--h', type=int, default=16,
                    help='Attention head num of transformer model.')

    parser.add_argument('--n_generator_layers', type=int, default=2,
                    help='Layer num of generator(MLP) model')

    parser.add_argument('--weight_split', type=str, default='1,1,1',
                    help='Unnormalized weights of Self-Attention/Adjacency/Distance Matrix respectively in MAT.')

    parser.add_argument('--leaky_relu_slope', type=float, default=0.0,
                    help='Leaky ReLU slope for activation functions.')

    parser.add_argument('--dense_output_nonlinearity',type=str,default='relu',
                    help='Activation Function for predict module, relu/tanh/none.')

    parser.add_argument('--distance_matrix_kernel',type=str,default='softmax',
                    help='Kernel applied on Distance Matrix, softmax/exp. For example, exp means setting D(i,j) of node i,j with distance d by exp(-d)')

    parser.add_argument('--dropout', type=float, default=0.1,
                    help='Dropout ratio.')

    parser.add_argument('--aggregation_type', type=str, default='mean',
                    help='Type for aggregeting node feature into graph feature, mean/sum/dummy_node.')

    parser.add_argument('--use_global_feature', action='store_true',
                    help='Whether to use global features(graph-level features).')

    parser.add_argument('--use_ffn_only', action='store_true',
                    help='Use DNN Generator which only considers global features. ')
    
    parser.add_argument('--d_mid_list', type=str, default='128,512',
                    help='Projection Layers to augment global feature dim to local feature dim.')

    parser.add_argument('--warmup_step', type=int, default=2000,
                    help='Warmup steps.')

    parser.add_argument('--epoch', type=int, default=100,
                    help='Epoch num.')  

    parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size.')   

    parser.add_argument('--fold', type=int, default=10,
                    help='Fold num.')    

    parser.add_argument('--lr', type=float, default=0.0007,
                    help='Maximum learning rate, (warmup_step * d_model) ** -0.5 .')

    

    

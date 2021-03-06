# MOFNet
MOFNet is a deep learning model that can predict adsorption isotherm for MOFs based on hierarchical representation, graph transformer and pressure adaptive mechanism. We elaborately design a hierarchical representation to describe the MOFs structure. A graph transformer is used to capture atomic level information, which can help learn chemical features required at low-pressure conditions. A pressure adaptive mechanism is used to interpolate and extrapolate the given limited data points by transfer learning, which can predict adsorption isotherms on a wider pressure range by only one model. The following is the architecture of MOFNet.

<img src="https://github.com/Matgen-project/MOFNet/blob/main/image/Fig1.jpg" width="100%">

## Installation
Please see dependencies in requirements.txt

## Dataset

We released the training and testing data on the [Matgen website](https://matgen.nscc-gz.cn/dataset.html), which can be obtained by the following command.
```
$ wget https://matgen.nscc-gz.cn/dataset/download/CSD-MOFDB_xx.tar.gz #xx: realased data
$ wget https://matgen.nscc-gz.cn/dataset/download/NIST-ISODB_xx.tar.gz
```

You can construct the data directory from the downloaded data as follows.

```
|-- data
||-- CSD-MOFDB
||-- NIST-ISODB
```

## CSD-MOFDB
We collected 7306, 6998 and 8562 MOFs for N<sub>2</sub>, CO<sub>2</sub> and CH<sub>4</sub> from the Cambridge Structural Database (CSD, version 5.4) dataset. 
GCMC simulations were carried out to calculate the adsorption data of MOFs for  N<sub>2</sub>, CO<sub>2</sub> and CH<sub>4</sub> using RASPA software. 
We set 8 pressure points from the range of 0.2 kPa - 80 kPa, 5 kPa – 20,000 kPa and 100 kPa – 10,000kPa for  N<sub>2</sub>, CO<sub>2</sub> and CH<sub>4</sub>, respectively.
```
| --CSD-MOFDB
||--CIFs  # CIF format files.
||--global_features  
||--label_by_GCMC  #calculated adsorption data by GCMC method.
||--local_features  
||--mol_unit   #molecule unit in mol format
||--README
```

## NIST-ISODB
We obtained 54 MOFs with 1876 pressure data points covering  N<sub>2</sub>, CO<sub>2</sub> and CH<sub>4</sub> adsorbate molecules from the NIST/ARPA-E database.

```
|--NIST-ISODB
||--CIFs   #CIF format files.
||--global_features  
||--isotherm_data  #experimental data.
||--local_features  
||--MOFNet   #MOFNet predicting results.
||--mol_unit  #molecule unit in mol format
||--README
```


## Processing

### How to generate local features?
First, the CSD package need to install on your server and use CSD Python API to obtain CIF files. We create a script in process file, and run the following command to generate local features file.
```
$ python process/process_csd_data.py <CSD_code>
```

### How to obtain global features?
The important structural properties including largest cavity diameter (LCD),pore-limiting diameter (PLD), and helium void fraction, etc., were calculated using open-source software Zeo++. 


## Model training
```
$ python -u train_mofnet.py --data_dir data/CSD-MOFDB --gas_type <gas_type> --pressure <pressure> --save_dir <save_dir_single> --use_global_feature
```

## Transfer learning
```
$ python -u pressure_adapt.py --data_dir data/CSD-MOFDB --gas_type <gas_type> --pressure <pressure> --save_dir <save_dir_all> --ori_dir <save_dir_single>/<gas_type>_<pressure> --adapter_dim 8
```

## Prediction
```
$ python -u nist_test.py --data_dir data/NIST-ISODB --gas_type <gas_type> --pressure <pressure> --save_dir <save_dir_all> --img_dir <img_dir>
```

We also welecome users to use our [3DStructGen UI interface](https://matgen.nscc-gz.cn/3dstructgen/v2/mod/3dstructgen_newUI.html) to predict crystal properties by the following steps:
```
# Upload your CIF crystal files into 3DStuctGen interface;
# Click "Caculate" button and use the APP of "Artificical Intelligence - MOF"
# Choose the uptake gas and pressure range you want to calculate and then submit.
```
<img src="https://github.com/Matgen-project/MOFNet/blob/3a97a3a487d72b5387e66c5651b5235eb85125da/image/3dstructgen-mof.png" width=100%>


## Acknowledgments
The implementation of the Graph Transformer module is built upon [Molecule Attention Transformer](https://github.com/ardigen/MAT).

## Reference:
[1]. Maziarka, {\L}ukasz and Danel, Tomasz and Mucha, S{\l}awomir and Rataj, Krzysztof and Tabor, Jacek and Jastrz{\k{e}}bski, Stanis{\l}aw: Molecule attention transforme. arXiv preprint arXiv:2002.08264 2020

[2]. Pin Chen, Yu Wang, Hui Yan, Sen Gao, Zexin Xu, Yangzhong Li, Qing Mo, Junkang Huang, Jun Tao, GeChuanqi Pan, Jiahui Li & Yunfei Du. 3DStructGen: an interactive web-based 3D structure generation for non-periodic molecule and crystal. J Cheminform 12, 7 (2020). https://doi.org/10.1186/s13321-020-0411-2

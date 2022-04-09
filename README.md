# MOFNet
MOFNet is a deep learning model that can predict adsorption isotherm for MOFs based on hierarchical representation, graph transformer and pressure adaptive mechanism. We elaborately design a hierarchical representation to describe the MOFs structure. A graph transformer is used to capture atomic level information, which can help learn chemical features required at low-pressure conditions. A pressure adaptive mechanism is used to interpolate and extrapolate the given limited data points by transfer learning, which can predict adsorption isotherms on a wider pressure range by only one model. The following is the the architecture of MOFNet.

<img src="https://github.com/Matgen-project/MOFNet/blob/main/image/Fig1.png" width="70%">

## Installation
Please see dependencies in requirements.txt

## Dataset

We released the training and testing data on the Matgen website, which can be obtained by the following command.
```
$ wget https://matgen.nscc-gz.cn/<dataset>
```

You can construct the data directory from the downloaded data as follows.

```
|-- data
||-- CSD-MOFDB
||-- NIST-ISODB
```

## CSD-MOFDB
We collected 7306, 6998 and 8562 MOFs for N2, CO2 and CH4 from the Cambridge Structural Database (CSD, version 5.4) dataset. 
GCMC simulations were carried out to calculate the adsorption data of MOFs for CO2, N2 and CH4 using RASPA software. 
We set 8 pressure points from the range of 0.2 kPa - 80 kPa, 5 kPa – 20,000 kPa and 100 kPa – 10,000kPa for N2, CO2 and CH4, respectively.
```
| --CSD-MOFDB
||--CIFs  
||--global_features  
||--label_by_GCMC  
||--local_features  
||--mol_unit  
||--README
```

## NIST-ISODB
We obtained 54 MOFs with 1876 pressure data points covering N2, CO2 and CH4 adsorbate molecules from the NIST/ARPA-E database.
| Gas type| MOFs name   | Data source |
| :---        | :----   | :--- |
| N2 |CPO-27-Fe	|10.1016j.micromeso.2011.12.035.isotherm2.json |
| N2 |Cu-MOF	|10.1016j.micromeso.2011.01.005.Isotherm3.json |
| N2 |HNUST-3	|10.1021Cg401180r.Isotherm2.json |
| N2 |MOF-205	|10.1039C4dt02300e.Isotherm4.json |
| N2 |NJU-Bai3	|10.1039C2cc16231h.Isotherm7.json |
| N2 |PCN-922	|10.1021Ic3019937.isotherm1.json |
| CO2 |CPF-1	|10.1039c1cc14836b.isotherm3.json |
| CO2 |HNUST-4	|10.1039C4ce01165a.Isotherm8.json |
| CO2 |JLU-Liu2	|10.1039C3dt52509k.Isotherm1.json |
| CO2 |MOF-505	|10.3969j.issn.1001-4861.2013.00.200.isotherm10.json |
| CO2 |NJU-Bai3	|10.1039C2cc16231h.Isotherm1.json |
| CO2 |UiO-67	|10.1016j.micromeso.2015.05.030.Isotherm9.json |
| CH4 |CPO-27-Mg	|10.1039C3sc51319j.isotherm47.json |
| CH4 |MOF-74-Ni	|10.1016j.ces.2014.12.001.Isotherm7.json |
| CH4 |MOF-205	|10.1126science.1192160.isotherm15.json |
| CH4 |Y-ftw-MOF-2	|10.1039c5sc00614g.Isotherm6.json |
| CH4 |ZIF-70	|10.1007s1093401500604.Isotherm33.json |
| CH4 |NJU-Bai3	|10.1039C2cc16231h.Isotherm14.json |


## Processing

### How to generate local features?
First, the CSD package need to install on your server and use CSD Python API to obtain CIF files. We create a script in utils files, and run the following command to generate local features file.
```
$ python process_csd_data.py <CSD_code>
```

### How to obtain global features?
The important structural properties including largest cavity diameter (LCD),pore-limiting diameter (PLD), and helium void fraction, etc., were calculated using open-source software Zeo++. 


## Model training
```
$ python -u train_mofnet.py --data_dir <data_dir> --gas_type <gas_type> --pressure <pressure> --save_dir <save_dir_single> --use_global_feature
```

## Transfer learning
```
python -u pressure_adapt.py --data_dir <data_dir> --gas_type <gas_type> --pressure <pressure> --save_dir <save_dir_all> --ori_dir <save_dir_single>/<gas_type>_<pressure> --adapter_dim 8
```

## Prediction
```
$ python -u nist_test.py --data_dir <data_dir> --gas_type <gas_type> --pressure <pressure> --save_dir <save_dir_all> --img_dir <img_dir>
```



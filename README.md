# MOFNet
MOFNet is a deep learning Model that can predict adsorption isotherm for MOFsbased on hierarchical representation, graph transformer and pressureadaptive mechanism. We elaborately design a hierarchical representationto describe the MOFs structure. A graph transformer is used to cap-ture atomic level information, which can help learn chemical featuresrequired at low-pressure conditions. A pressure adaptive mechanism is used to interpolate and extrapolate the given limited data points bytransfer learning, which can predict adsorption isotherms on a widerpressure range by only one model. The following is the the architecture of MOFNet.

<img src="https://github.com/Matgen-project/MOFNet/blob/main/image/Fig1.png" width="50%">

## Installation
Please see dependencies in requirements.txt

## Dataset
## CSD-MOFDB
We released the trained data on the Magen website, which can be obtained by the following command.
```
$ wget https://matgen.nscc-gz.cn/xx
```
## NIST-ISODB
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

## Process
### How to generate local features?
First, the CSD package need to install on your server and use CSD Python API to obtain CIF files. We create a script in utils files, and run the following command to generate local features file.
```
$ python process_data.py $CSD_code
```

### How to obtain global features?
The important structural properties including largest cavity diameter (LCD),pore-limiting diameter (PLD), and helium void fraction, etc., were calculatedusing  open-source software Zeo++. 


## Training
```
$ python -u train_fold.py --dense_output_nonlinearity silu --distance_matrix_kernel bessel --epoch 3 --batch_size 64 --data_dir ./data --gas_type $1 --pressure $2 --save_dir ./test/mof_rbf --use_global_feature
```

## Transfer learning
```
python -u pressure_adapt.py --data_dir ./data --gas_type $1 --pressure $2 --epoch 300 --save_dir ./mof_adapted_rbf --ori_dir ./mof_model_rbf/v3/$1_$2 --lr 0.0007 --adapter_dim 8
```

## Prediction
```
$ python -u real_test.py --data_dir 287-Cu-MOF --gas_type CH4 --pressure 5e4 --save_dir ./mof_adapted_rbf
```



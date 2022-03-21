# MOFNet
Deep learning model for predicting adsorption isotherms of MOFs

## Installation
Please see dependencies in requirements.txt

## Dataset
## CSD-MOFDB
We released the trained data on the Magen website, which can be obtained by the following command.
```
$wget https://matgen.nscc-gz.cn/
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
···
python process_data.py $CSD_code
···

### How to obtain global features?
The important structural properties including largest cavity diameter (LCD),pore-limiting diameter (PLD), and helium void fraction, etc., were calculatedusing  open-source software Zeo++. 


## Training

## Prediction


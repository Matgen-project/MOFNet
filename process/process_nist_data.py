import json
import os
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser

t_dict = {77:"Nitrogen", 273:"Carbon Dioxide", 298:"Methane"}
unit_dic = {"mmol/g":1, "mol/g":0.001, 'mmol/kg':1000}
m_dic = {"Nitrogen":28.0134, "Methane":16.0424, "Carbon Dioxide":44.0094}
def get_unit_factor(unit,ads):
    if unit in unit_dic:
        return 1 / unit_dic[unit]
    elif unit == "cm3(STP)/g":
        return 1 / 22.4139
    elif unit == 'mg/g':
        return 1 / m_dic[ads]
    else:
        return None

def norm_str(ori):
    ori = ori.split('.')[0].split('-')
    if ori[-1] == 'clean':
        ori = ori[:-1]
    elif ori[-2] == 'clean':
        ori = ori[:-2]
    return '-'.join(ori[1:])

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                help='NIST data directory.')
    args = parser.parse_args()
    prefix = os.path.join(args.data_dir,'isotherm_data')
    pres_all = {"CH4":{"num":0, "data":[]}, "CO2":{"num":0, "data":[]}, "N2":{"num":0, "data":[]}}
    for gas_type in ['CH4','CO2','N2']:
        gas_pref = os.path.join(prefix, gas_type)
        files = os.listdir(gas_pref)
        for js in tqdm(files):
            with open(os.path.join(gas_pref, js), "r") as f:
                dic = json.load(f)
            name = dic['adsorbent']['name']
            t = dic['temperature']
            if t not in t_dict:
                continue
            tar_obj = t_dict[t]
            unit_factor = get_unit_factor(dic['adsorptionUnits'], tar_obj)
            if not unit_factor:
                continue
            tar_key = None
            for ads in dic['adsorbates']:
                if ads['name'] == tar_obj:
                    tar_key = ads['InChIKey']
                    break
            if not tar_key:
                continue
            pres_ret = []
            for d in dic['isotherm_data']:
                pres = d['pressure'] * 1e5
                for sd in d['species_data']:
                    if sd['InChIKey'] == tar_key:
                        tar_abs = sd['adsorption'] * unit_factor
                pres_ret.append({'pressure':pres, 'adsorption':tar_abs})
            pres_all[gas_type]['num'] += 1
            pres_all[gas_type]['data'].append({"name":name, "filename":js, "isotherm_data":pres_ret})
    with open(os.path.join(prefix,'all.json'),'w') as f:
        json.dump(pres_all, f)
    
        
import json
import os
import pandas as pd
from tqdm import tqdm

t_dict = {77:"Nitrogen", 273:"Carbon Dioxide", 298:"Methane"}
# CH4: 1 mol/kg = 16.0424milligram/gram = 1.7576276757 cm^3 (STP)/gr = 39.9071 cm^3 (STP)/cm^3
# N2: 1 mol/kg = 28.0134 milligram/gram = 22.4139 cm^3 (STP)/gr = 22.8617 cm^3 (STP)/cm^3 
# CO2:  1 mol/kg = 44.0094 mol/kg framework = 22.4139 cm^3 (STP)/gr = 29.4549 cm^3 (STP)/cm^3
# unit_dic = {"mmol/g":1, 'mg/g':16.0424, 'cm3(STP)/g':1.7576276757, "mol/g":0.001, 'g/g':0.0160424, 'mmol/kg':1000}
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
        # print(unit)
        return None

def norm_str(ori):
    ori = ori.split('.')[0].split('-')
    if ori[-1] == 'clean':
        ori = ori[:-1]
    elif ori[-2] == 'clean':
        ori = ori[:-2]
    return '-'.join(ori[1:])

if __name__ == "__main__":
    prefix = 'data/exp_data/label_nist/data/'
    # files = os.listdir(prefix)
    # pres_all = {"Methane":{"num":0, "data":[]}, "Carbon Dioxide":{"num":0, "data":[]}, "Nitrogen":{"num":0, "data":[]}}
    # for js in tqdm(files):
    #     with open(prefix + js, "r") as f:
    #         dic = json.load(f)
    #     name = dic['adsorbent']['name']
    #     t = dic['temperature']
    #     if t not in t_dict:
    #         continue
    #     tar_obj = t_dict[t]
    #     unit_factor = get_unit_factor(dic['adsorptionUnits'], tar_obj)
    #     if not unit_factor:
    #         continue
    #     tar_key = None
    #     for ads in dic['adsorbates']:
    #         if ads['name'] == tar_obj:
    #             tar_key = ads['InChIKey']
    #             break
    #     if not tar_key:
    #         continue
    #     pres_ret = []
    #     for d in dic['isotherm_data']:
    #         pres = d['pressure'] * 1e5
    #         for sd in d['species_data']:
    #             if sd['InChIKey'] == tar_key:
    #                 tar_abs = sd['adsorption'] * unit_factor
    #         pres_ret.append({'pressure':pres, 'adsorption':tar_abs})
    #     pres_all[tar_obj]['num'] += 1
    #     pres_all[tar_obj]['data'].append({"name":name, "filename":js, "isotherm_data":pres_ret})
    # with open(prefix + '../all.json','w') as f:
    #     json.dump(pres_all, f)
    with open(prefix + '../all.json') as f:
        pres_all = json.load(f)
    data_df = pd.read_csv('data/exp_data/global_feat/exp_geo_all.csv', header=0)
    data_x = data_df['name'].values
    for g in pres_all.keys():
        names = [_['name'] for _ in pres_all[g]['data']]
        ans = []
        for n in tqdm(data_x):
            if norm_str(n) in names:
                ans.append(n+'.p')
        with open(f'data/exp_data/{g}_list','w') as f:
            f.write('\n'.join(ans))
    
        
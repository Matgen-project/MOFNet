from turtle import width
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas

tar_gas = 'CH4'
tar_pres = ['5e4', '2e5', '100e5']
# tar_gas = 'CO2'
# tar_pres = ['1e4', '5e5', '5e7']
# tar_gas = 'N2'
# tar_pres = ['2e2', '2e4', '1e5']

with open(f'/GPUFS/nscc-gz_material_13/projects/MOFNet/mof_adapted_rbf/{tar_gas}_{tar_pres[0]}/shap_result_{tar_pres[0]}.p', 'rb') as f:
    p1 = pickle.load(f)
ans1 = p1[tar_pres[0]].mean(axis=0) / p1[tar_pres[0]].mean(axis=0).sum()

with open(f'/GPUFS/nscc-gz_material_13/projects/MOFNet/mof_adapted_rbf/{tar_gas}_{tar_pres[0]}/shap_result_{tar_pres[1]}.p', 'rb') as f:
    p2 = pickle.load(f)
ans2 = p2[tar_pres[1]].mean(axis=0) / p2[tar_pres[1]].mean(axis=0).sum()

with open(f'/GPUFS/nscc-gz_material_13/projects/MOFNet/mof_adapted_rbf/{tar_gas}_{tar_pres[0]}/shap_result_{tar_pres[2]}.p', 'rb') as f:
    p3 = pickle.load(f)
ans3 = p3[tar_pres[2]].mean(axis=0) / p3[tar_pres[2]].mean(axis=0).sum()
 
name_list = ['Local','Density','PLD','LCD','VSA','GSA','voidfract','pore_vol','chan_num','oms_per_cell'][::-1]
ans1 = ans1[::-1]
ans2 = ans2[::-1]
ans3 = ans3[::-1]

x = np.arange(len(name_list))
width = 0.25

plt.barh(x + width, ans1, width, tick_label = name_list, label = 'low pressure')
plt.barh(x, ans2, width, tick_label = name_list, label = 'med. pressure')
plt.barh(x - width, ans3, width, tick_label = name_list, label = 'high pressure')
plt.legend(loc='best')
# plt.title('Relative Feature Importance')
plt.tight_layout()
plt.savefig(f'images/others/shap_{tar_gas}.png')

tar_dic = {'low':ans1, 'med.':ans2, 'high':ans3}
df = pandas.DataFrame.from_dict(tar_dic, orient='index',columns=name_list)
df.to_csv(f'images/others/shap_{tar_gas}.csv')


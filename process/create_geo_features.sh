#!/bin/sh

data_dir=${root_dir_for_cifs}/
cell_num=$2
i=$1
name=`echo $i |cut -d '.' -f 1`
#argument 1 and 2
#../network -ha -res ~/wsl/work/clean/$i >/dev/null
~/bin/network -ha -res ${data_dir}/${i} >/dev/null
LCD=`head -n 1  ${data_dir}/${name}.res | awk '{print $4}'`
PLD=`head -n 1  ${data_dir}/${name}.res | awk '{print $3}'`
#exit
rm  ${data_dir}/${name}.res
#argument 3 and 4 and 5
#../network -ha -sa 1.86 1.86 2000 ~/wsl/work/clean/$i >/dev/null
~/bin/network -ha -sa 1.86 1.86 2000 ${data_dir}/${i} >/dev/null
VSA=`head -n 1  ${data_dir}/${name}.sa | awk '{print $10}'`
GSA=`head -n 1  ${data_dir}/${name}.sa | awk '{print $12}'`
Density=`head -n 1  ${data_dir}/${name}.sa | awk '{print $6}'`
chan_num_sa=`sed -n '2p'  ${data_dir}/${name}.sa | awk '{print $2}'`
rm  ${data_dir}/${name}.sa
#argument 6 and 7  
#   ../network -ha -vol 0 0 50000 ~/wsl/work/clean/$i >/dev/null
~/bin/network -ha -vol 0 0 50000 ${data_dir}/${i} >/dev/null
voidfract=`head -n 1  ${data_dir}/${name}.vol | awk '{print $10}'` 
porevolume=`head -n 1  ${data_dir}/${name}.vol | awk '{print $12}'`
rm  ${data_dir}/${name}.vol

~/bin/network -oms /tmp/${i}.cif >/dev/null
oms=`tail -n 1  /tmp/${i}.oms | awk '{print $3}'`
rm /tmp/${i}.oms

chan_num_sa=`echo "scale=6;${chan_num_sa}/${cell_num}" | bc`
porevolume=`echo "scale=6;${porevolume}/${cell_num}" | bc`
oms=`echo "scale=6;${oms}/${cell_num}" | bc`
#printf "%-20s%-10s%-10s%-10s%-10s%-10s%-10s%-15s%-15s\n" $name $Density $PLD $LCD $VSA $GSA $voidfract $porevolume $chan_num_sa $oms
echo "$i,$LCD,$PLD,$VSA,$GSA,$Density,$voidfract,$porevolume,$chan_num_sa,$oms" 

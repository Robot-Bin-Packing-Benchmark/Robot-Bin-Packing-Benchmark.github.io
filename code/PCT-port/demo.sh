#!/bin/bash2

settings="1"
datas="time_series occupancy flat_long"
methods="HeightmapMin LSAH MACS RANDOM OnlineBPH DBL BR SDFPack PCT"
test_data_configs="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29"
#test_data_configs="0 1"
# --setting 1 --data time_series --method PCT --test_data_config 0

for setting in ${settings}
do
  echo $setting running!
  for data in ${datas}
  do
    echo $data running!
    for method in ${methods}
    do
      echo $method running!
      for test_data_config in ${test_data_configs}
      do
        echo $test_data_config running!
        python main.py --setting $setting --data $data --method $method --test_data_config $test_data_config
      echo $method done!
      done
    echo $data done!
    done
  echo $setting done!
  done
done
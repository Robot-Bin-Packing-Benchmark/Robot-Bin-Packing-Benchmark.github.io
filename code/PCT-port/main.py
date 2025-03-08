# -*- coding=utf-8 -*-
import json

from tools import get_args, load_data
from pack import pack

'''
    setting1: 不考虑物理仿真，不考虑机械臂 (只考虑geometry 也就是直接静态放置)
'''

if __name__ == '__main__':
    args, args_base = get_args()

    sizes = load_data(args.data, args.test_data_config, args_base)
    actions, planning_times = pack(args, args_base, sizes)

    with open("action.json", "w") as f:
        json.dump(actions, f)
    with open("planning_time.json", "w") as f:
        json.dump(planning_times, f)

    print(actions)
    print(planning_times)
    print("end!")

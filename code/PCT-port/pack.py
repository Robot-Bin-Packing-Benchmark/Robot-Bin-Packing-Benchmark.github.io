# -*- coding=utf-8 -*-
import torch
import warnings
warnings.filterwarnings("ignore")
from pct_envs.PctDiscrete0 import PackingDiscrete
from packer.app import pack_box, get_policy

def pack(args, args_base, sizes):

    policy, infos = get_policy(args)

    if args.setting == 1:
        check_stability = False
    else:
        check_stability = True

    # 导入 packing 理想环境
    env = PackingDiscrete(setting=1, check_stability=check_stability,
                          container_size=args_base.Scene.target_container_size,
                          item_set=sizes, block_unit=args_base.Scene.block_unit, load_test_data=True,
                          args=infos['args'])
    env.space.policy = policy

    obs = env.reset()
    obs = torch.FloatTensor(obs).unsqueeze(dim=0)

    actions = []
    planning_times = []

    print("------------------ Start packing ------------------")
    for n in range(len(sizes)):
        env.next_item_ID = n + 1
        placeable, action, infos, planning_time = pack_box(env, infos, obs, args.method, policy)

        if not placeable:
            print(f"{n} boxes were successfully packed!")
            break
        else:
            actions.append(action)
            planning_times.append(planning_time)

        obs = infos['next_obs']
        if obs is None:
            pass
        else:
            obs = torch.FloatTensor(obs).unsqueeze(dim=0)

    return actions, planning_times

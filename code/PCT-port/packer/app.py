from packer.pct_model.model import DRL_GAT
from packer.pct_model.tools import *
from tools import load_config
import time


def get_policy(args):
    # LSAH   MACS   RANDOM  OnlineBPH   DBL  BR  MTPE IR_HM BLBF FF
    infos = dict()
    infos['args'] = None
        
    if args.method == 'PCT':
        # args = get_pct_args()
        # infos['args'] = args
        args_method = load_config(args.config_learning_method)

        if args.data == 'random' or args.data == 'time_series':
            if args.setting == 1:
                args_method.model_path = args_method.model_path_random_time_series
            else:
                args_method.model_path = args_method.model_path_random_time_series_pybullet
        elif args.data == 'occupancy':
            if args.setting == 1:
                args_method.model_path = args_method.model_path_occupancy
            else:
                args_method.model_path = args_method.model_path_occupancy_pybullet
        elif args.data == 'flat_long':
            if args.setting == 1:
                args_method.model_path = args_method.model_path_flat_long
            else:
                args_method.model_path = args_method.model_path_flat_long_pybullet

        infos['args'] = args_method
        PCT_policy = DRL_GAT()

        # Load the trained model
        model_path = args_method.model_path
        PCT_policy = load_policy(model_path, PCT_policy)
        print('Pre-train model loaded successfully!')
        PCT_policy.eval()
        
    return PCT_policy, infos



def pct(PCT_policy, env, obs, args, eval_freq = 100, factor = 1):
    all_nodes, leaf_nodes = get_leaf_nodes_with_factor(obs, 1, args.internal_node_holder, args.leaf_node_holder)
    batchX = torch.arange(1)

    with torch.no_grad():
        selectedlogProb, selectedIdx, policy_dist_entropy, value = PCT_policy(all_nodes, True, normFactor=factor)
    selected_leaf_node = leaf_nodes[batchX, selectedIdx.squeeze()]      # tensor([[ 8.,  0.,  0., 10.,  5., 10.,  0.,  0.,  1.]], device='cuda:0')

    action = selected_leaf_node.cpu().numpy()[0][0:6]
    now_action, box_size = env.LeafNode2Action(action)
    
    # check rot
    init_box_size = env.next_box
    if box_size[0] == init_box_size[0] and box_size[1] == init_box_size[1]:
        rot = 0
    else:
        rot = 1
    rec = env.space.plain[now_action[1]:now_action[1] + box_size[0], now_action[2]:now_action[2] + box_size[1]]
    lz = np.max(rec)
    
    obs, reward, done, infos = env.step(action)
    new_action = (rot,) + now_action[1:]
    
    return done, new_action, lz, obs



def pack_box(env, infos, obs, method, policy):
    # LSAH   MACS   RANDOM  OnlineBPH   DBL  BR
    if method == 'PCT':
        args = infos['args']
        normFactor = 1.0 / np.max(env.bin_size)
        start_time = time.time()
        done, action, lz, next_obs = pct(policy, env, obs, args, eval_freq=100, factor=normFactor)
        end_time = time.time()
        planning_time = end_time - start_time

        placeable = not done
        if placeable == False:
            return placeable, [], infos, planning_time
        rotation_flag, lx, ly = action[0], action[1], action[2]

        infos['next_obs'] = next_obs

    return placeable, action, infos, planning_time

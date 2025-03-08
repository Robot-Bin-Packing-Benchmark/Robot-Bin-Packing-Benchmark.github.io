import numpy as np
import pybullet
import torch
import yaml
import argparse
from easydict import EasyDict as ed


def get_args():
    parser = argparse.ArgumentParser(description="training vae configure")
    parser.add_argument("--config", help="Base configure file name", type=str, default='configs/base.yaml')
    parser.add_argument("--setting", help="setting1 is no consideration of physical simulation, no consideration of robotic arms\
                                                         setting2 is consideration of physical simulation, no consideration of robotic arms\
                                                         setting3 is consideration of physical simulation, consideration of robotic arms", type=int, default=1)
    parser.add_argument("--is_3d_packing", help="3d packing or 2d packing, 1 or 0", type=int, default=1)
    parser.add_argument("--data", help="regular data: random, time_series, occupancy, flat_long.", type=str, default='occupancy')
    parser.add_argument("--method", help="regular method: LeftBottom, HeightmapMin, LSAH, MACS, RANDOM, OnlineBPH, DBL, BR, SDFPack, PCT, PackE.", type=str, default='PCT')
    parser.add_argument("--config_learning_method", help="Test learning method configure file name: pct.yaml", type=str, default='configs/pct.yaml')
    parser.add_argument("--test_data_config", help="test data configuration, 0 is the default data, the range is 0-29", type=int, default=0)

    args = parser.parse_args()
    args_base = load_config(args.config)
    
    args_base = get_data_info(args, args_base)

    return args, args_base


def load_config(config_dir, easy=True):
    cfg = yaml.load(open(config_dir), yaml.FullLoader)
    if easy is True:
        cfg = ed(cfg)
    return cfg


def get_data_info(args, args_base):
   # 3d regular
    if args.data == 'random':
        container_size = args_base.Data.Random.container_size
        easy_container_size = [container_size[0], container_size[1], min(container_size[0], container_size[1])]
        args_base.Scene.target_container_size = easy_container_size
    elif args.data == 'time_series':
        args_base.Scene.target_container_size = args_base.Data.Time_series.container_size
    elif args.data == 'occupancy':
        args_base.Scene.target_container_size = args_base.Data.Occupancy.container_size
    elif args.data == 'flat_long':
        args_base.Scene.target_container_size = args_base.Data.Flat_long.container_size

    args_base.method = args.method


    return args_base


def pose_to_mat(pose):
  pos, quat = pose
  mat = np.identity(4)

  mat[:3,:3] = np.array(pybullet.getMatrixFromQuaternion(quat)).reshape(3,3)
  mat[:3,3] = pos
  return mat

def load_data(data, data_config, args):
    if data == 'random':
        sizes = np.load(args.Data.Random.data_path, allow_pickle=True )

    elif data == 'time_series':
        file_path = args.Data.Time_series.data_path + '/data_time_series_' + str(data_config) + '.pt'
        data = torch.load(file_path)
        sizes = np.array(data['data'])

    elif data == 'occupancy':
        file_path = args.Data.Occupancy.data_path + '/data_occupancy_' + str(data_config) + '.pt'
        data = torch.load(file_path)
        sizes = np.array(data['data'])

    elif data == 'flat_long':
        file_path = args.Data.Flat_long.data_path + '/data_flat_long_' + str(data_config) + '.pt'
        data = torch.load(file_path)
        sizes = np.array(data['data'])


    pack_init_sizes = np.zeros_like(sizes)  # 创建一个相同形状的数组存储结果
    pack_init_sizes[:, :2] = np.ceil(sizes[:, :2] / args.Scene.block_unit).astype(int)  # 前两列取整
    pack_init_sizes[:, 2] = sizes[:, 2] / args.Scene.block_unit  # 最后一列不取整
    pack_init_sizes[:, -1] = np.round(pack_init_sizes[:, -1], 1)
    pack_init_sizes = pack_init_sizes[pack_init_sizes.sum(axis=1) != 0]
    pack_init_sizes = pack_init_sizes[:, (pack_init_sizes.sum(axis=0) != 0)]


    return pack_init_sizes

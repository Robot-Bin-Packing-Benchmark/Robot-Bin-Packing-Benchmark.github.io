import numpy as np
import torch
from .space import Space

import gym
from .binCreator import RandomBoxCreator, LoadBoxCreator, newLoadBoxCreator, BoxCreator
import random



class PackingDiscrete(gym.Env):
    def __init__(self,
                 setting,
                 check_stability,
                 threshold=0.01,
                 container_size=(10, 10, 10),
                 item_set=None, target_origin=np.array([0.35, 0.28, 0.025]),block_unit=0.03, load_test_data=False, data_name=None, 
                 internal_node_holder=80, leaf_node_holder=50, next_holder=1, shuffle=False,
                 LNES = 'EMS', args=None, selectedAction=500,
                 **kwags):

        self.check_stability = check_stability
        self.internal_node_holder = internal_node_holder    # 内部节点的容器大小
        self.leaf_node_holder = leaf_node_holder    # 叶子节点的容器大小
        self.next_holder = next_holder  # 下一个箱子信息的容器大小

        self.shuffle = shuffle  # 是否在生成叶子节点后对它们进行洗牌
        self.bin_size = container_size  # 容器的尺寸，格式为 (长, 宽, 高)
        self.size_minimum = np.min(np.array(item_set))  # 箱子集合中最小的尺寸
        self.setting = setting
        self.item_set = item_set    # 可用箱子的尺寸集合
        self.target_origin = target_origin
        self.block_unit = block_unit
        self.threshold = threshold
        
        if self.setting == 2: self.orientation = 6  #  箱子的放置方向数
        else: self.orientation = 2
        
        # The class that maintains the contents of the bin.
        self.space = Space(*self.bin_size, self.size_minimum, self.internal_node_holder, self.target_origin, self.block_unit, self.item_set, self.check_stability, self.threshold)

        # Generator for train/test data
        if not load_test_data:
            assert item_set is not None
            self.box_creator = RandomBoxCreator(item_set)   # 用于生成箱子的尺寸
            assert isinstance(self.box_creator, BoxCreator)
        if load_test_data:
            # self.box_creator = LoadBoxCreator(data_name)
            self.box_creator = newLoadBoxCreator(item_set)

        self.test = load_test_data
        self.observation_space = gym.spaces.Box(low=0.0, high=self.space.height,    # gym.spaces.Box调用的哪里？？？？？？？？？？？？
                                                shape=((self.internal_node_holder + self.leaf_node_holder + self.next_holder) * 9,))
        self.next_box_vec = np.zeros((self.next_holder, 9))

        self.LNES = LNES  # Leaf Node Expansion Schemes: EMS (recommend), EV, EP, CP, FC
        


    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            random.seed(seed)
            self.SEED = seed
        return [seed]

    # Calculate space utilization inside a bin.
    def get_box_ratio(self):
        coming_box = self.next_box
        return (coming_box[0] * coming_box[1] * coming_box[2]) / (self.space.plain_size[0] * self.space.plain_size[1] * self.space.plain_size[2])

    def reset(self):
        self.box_creator.reset()
        self.packed = []
        self.space.reset()
        self.box_creator.generate_box_size()
        cur_observation = self.cur_observation()
        return cur_observation

    # Count and return all PCT nodes.
    def cur_observation(self):
        boxes = []
        leaf_nodes = []
        self.next_box = self.gen_next_box()

        if self.test:
            if self.setting == 3: self.next_den = self.next_box[3]
            else: self.next_den = 1
            # self.next_box = [int(self.next_box[0]), int(self.next_box[1]), int(self.next_box[2])]
            self.next_box = [int(self.next_box[0]), int(self.next_box[1]), self.next_box[2]]
        else:
            if self.setting < 3: self.next_den = 1
            else:
                self.next_den = np.random.random()
                while self.next_den == 0:
                    self.next_den = np.random.random()

        boxes.append(self.space.box_vec)
        leaf_nodes.append(self.get_possible_position())

        next_box = sorted(list(self.next_box))
        self.next_box_vec[:, 3:6] = next_box
        self.next_box_vec[:, 0] = self.next_den
        self.next_box_vec[:, -1] = 1
        return np.reshape(np.concatenate((*boxes, *leaf_nodes, self.next_box_vec)), (-1))


    
    # Generate the next item to be placed.
    def gen_next_box(self):
        return self.box_creator.preview(1)[0]

    # Detect potential leaf nodes and check their feasibility.
    def get_possible_position(self):
        if   self.LNES == 'EMS':
            allPostion = self.space.EMSPoint(self.next_box,  self.setting)
        elif self.LNES == 'EV':
            allPostion = self.space.EventPoint(self.next_box,  self.setting)
        elif self.LNES == 'EP':
            allPostion = self.space.ExtremePoint2D(self.next_box, self.setting)
        elif self.LNES == 'CP':
            allPostion = self.space.CornerPoint(self.next_box, self.setting)
        elif self.LNES == 'FC':
            allPostion = self.space.FullCoord(self.next_box, self.setting)
        else:
            assert False, 'Wrong LNES'

        if self.shuffle:
            np.random.shuffle(allPostion)

        leaf_node_idx = 0
        leaf_node_vec = np.zeros((self.leaf_node_holder, 9))
        tmp_list = []

        for position in allPostion:
            xs, ys, zs, xe, ye, ze = position
            x = xe - xs
            y = ye - ys
            z = ze - zs

            if self.space.drop_box_virtual([int(x), int(y), z], (int(xs), int(ys)), False, self.next_den, self.setting):
                tmp_list.append([xs, ys, zs, xe, ye, self.bin_size[2], 0, 0, 1])
                leaf_node_idx += 1

            if leaf_node_idx >= self.leaf_node_holder: break

        if len(tmp_list) != 0:
            leaf_node_vec[0:len(tmp_list)] = np.array(tmp_list)

        return leaf_node_vec

    # Convert the selected leaf node to the placement of the current item.
    def LeafNode2Action(self, leaf_node):
        if np.sum(leaf_node[0:6]) == 0: return (0, 0, 0), self.next_box
        x = int(leaf_node[3] - leaf_node[0])
        y = int(leaf_node[4] - leaf_node[1])
        z = list(self.next_box)
        z.remove(x)
        z.remove(y)
        z = z[0]
        action = (0, int(leaf_node[0]), int(leaf_node[1]))
        # next_box = (x, y, int(z))
        next_box = (x, y, z)
        return action, next_box

    def step(self, action):
        if len(action) != 3: action, next_box = self.LeafNode2Action(action)
        else: next_box = self.next_box

        idx = [action[1], action[2]]
        bin_index = 0
        rotation_flag = action[0]
        succeeded = self.space.drop_box(next_box, idx, rotation_flag, self.next_den, self.setting, self.packed)

        if not succeeded:
            reward = 0.0
            done = True
            info = {'counter': len(self.space.boxes), 'ratio': self.space.get_ratio(),
                    'reward': self.space.get_ratio() * 10}
            return self.cur_observation(), reward, done, info

        ################################################
        ############# cal leaf nodes here ##############
        ################################################
        packed_box = self.space.boxes[-1]

        if  self.LNES == 'EMS':
            self.space.GENEMS([packed_box.lx, packed_box.ly, packed_box.lz,
                                           packed_box.lx + packed_box.x, packed_box.ly + packed_box.y,
                                           packed_box.lz + packed_box.z])

        self.packed.append(
            [packed_box.x, packed_box.y, packed_box.z, packed_box.lx, packed_box.ly, packed_box.lz, bin_index])

        box_ratio = self.get_box_ratio()
        self.box_creator.drop_box()  # remove current box from the list
        self.box_creator.generate_box_size()  # add a new box to the list
        reward = box_ratio * 10

        done = False
        info = dict()
        info['counter'] = len(self.space.boxes)

        if isinstance(self.space.policy, str):
            self.cur_observation()
            return None, reward, done, info
        else:
            return self.cur_observation(), reward, done, info



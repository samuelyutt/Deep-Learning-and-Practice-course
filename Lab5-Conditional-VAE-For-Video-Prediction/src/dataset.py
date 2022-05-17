from turtle import position
import torch
import os
import numpy as np
import csv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

default_transform = transforms.Compose([
    transforms.ToTensor(),
    ])

class bair_robot_pushing_dataset(Dataset):
    def __init__(self, args, mode='train', transform=default_transform, root='../data'):
        assert mode == 'train' or mode == 'test' or mode == 'validate'
        self.path = f'{root}/{mode}'
        self.trajs = []
        self.transform = transform
        self.seed_is_set = False
        self.n_frames = args.n_past + args.n_future

        trajs_dirs = os.listdir(self.path)
        for trajs_dir in trajs_dirs:
            if '.tfrecords' in trajs_dir:
                traj_dirs = os.listdir(f'{self.path}/{trajs_dir}')
                for traj_dir in traj_dirs:
                    self.trajs.append((trajs_dir, int(traj_dir)))

        if args.seed is not None:
            self.set_seed(args.seed)

        # raise NotImplementedError
        
    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        return len(self.trajs)
        # raise NotImplementedError

    def get_seq(self, index):
        seq = []
        trajs_dir, traj_dir = self.trajs[index]
        for i in range(self.n_frames):
            img = Image.open(f'{self.path}/{trajs_dir}/{traj_dir}/{i}.png')
            img = self.transform(img)
            seq.append(img)
        seq = np.stack(seq)
        return seq
        # raise NotImplementedError
    
    def get_csv(self, index):
        trajs_dir, traj_dir = self.trajs[index]
        actions = np.loadtxt(f'{self.path}/{trajs_dir}/{traj_dir}/actions.csv', delimiter=',')
        positions = np.loadtxt(f'{self.path}/{trajs_dir}/{traj_dir}/endeffector_positions.csv', delimiter=',')
        csv = np.concatenate((actions, positions), axis=1)[:self.n_frames]
        return csv
        # raise NotImplementedError
    
    def __getitem__(self, index):
        self.set_seed(index)
        seq = self.get_seq(index)
        cond =  self.get_csv(index)
        return seq, cond

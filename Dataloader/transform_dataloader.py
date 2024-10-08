"modelnet-C生成"
from random import sample
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import Sampler
import os
import h5py
# import open3d as o3d
from tqdm import tqdm
from torch.utils.data import DataLoader
from Dataloader.data_utils import *
import random
from torchvision import transforms
import pickle

np.random.seed(0)


class MiniImageNet(Dataset):
    def __init__(self, root, split='train', num_point=1024, aug=True, fold=0, data_aug=False):
        super().__init__()
        self.root = root
        self.split = split
        self.num_point = num_point
        self.aug = aug
        self.fold = fold
        self.data_aug = data_aug
        # self.sample_num = sample_num  # sample_num=5

        self.point_list, self.point_label = self.get_point()

        self.transforms = transforms.Compose(
            [
                PointcloudToTensor(),
                PointcloudScale(),
                PointcloudRotate(),
                PointcloudRotatePerturbation(),
                PointcloudTranslate(),
                PointcloudJitter(),
            ]
        )

        rand_points = np.random.uniform(-1, 1, 40000)
        x1 = rand_points[:20000]
        x2 = rand_points[20000:]
        power_sum = x1 ** 2 + x2 ** 2
        p_filter = power_sum < 1
        power_sum = power_sum[p_filter]
        sqrt_sum = np.sqrt(1 - power_sum)
        x1 = x1[p_filter]
        x2 = x2[p_filter]
        x = (2 * x1 * sqrt_sum).reshape(-1, 1)
        y = (2 * x2 * sqrt_sum).reshape(-1, 1)
        z = (1 - 2 * power_sum).reshape(-1, 1)
        self.density_points = np.hstack([x, y, z])
        self.fn = [
            lambda pc: drop_hole(pc, p=0.24),
            lambda pc: drop_hole(pc, p=0.36),
            lambda pc: drop_hole(pc, p=0.45),
            lambda pc: p_scan(pc, pixel_size=0.017),
            lambda pc: p_scan(pc, pixel_size=0.022),
            lambda pc: p_scan(pc, pixel_size=0.035),
            lambda pc: density(pc, self.density_points[np.random.choice(self.density_points.shape[0])], 1.3),
            lambda pc: density(pc, self.density_points[np.random.choice(self.density_points.shape[0])], 1.4),
            lambda pc: density(pc, self.density_points[np.random.choice(self.density_points.shape[0])], 1.6),
            lambda pc: pc.copy(),
        ]
        self.transfrom_num = len(self.fn)

    def get_point(self):
        # == will be returned later ==
        list_of_points = []
        list_of_labels = []
        # ============================

        picked_index = np.zeros(40)
        picked_index[self.fold * 10:(self.fold + 1) * 10] = 1  # 选择某一折作为测试集,将这一部分(10类)置为1

        class_list = np.arange(40)
        if self.split == 'train':  #
            picked_index = (1 - picked_index).astype(bool)  # 1部分变为0,0部分变为1成为训练集
        else:
            picked_index = picked_index.astype(bool)

        class_list = class_list[picked_index]  # 选中相应类别数
        for c in class_list:
            class_fold = os.path.join(self.root, str(c), '%d.npy' % (c))  # 第c类
            with open(class_fold, 'rb') as f:
                points = pickle.load(f)
            for i in range(len(points)):
                list_of_labels.append(c)
                list_of_points.append(points[i])

        return list_of_points, list_of_labels

    def translate_pointcloud(self, pointcloud):
        xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
        xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

        translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
        return translated_pointcloud

    def transform_pointcloud(self, pointcloud):  # (1024, 3)  # 用到fn
        trans_id = random.randint(0, self.transfrom_num - 1)
        trans_pointcloud = self.fn[trans_id](pointcloud)
        return trans_pointcloud

    def __getitem__(self, item):
        pointcloud = self.point_list[item][:self.num_point]
        label = self.point_label[item]
        pn = self.num_point
        if self.split == 'train':
            pointcloud = self.transform_pointcloud(pointcloud)
            if self.aug:
                pointcloud = self.transforms(pointcloud)  # 用到transforms
                pointcloud = pointcloud.numpy()
            pn = min(pointcloud.shape[0], self.num_point)
            if pn < self.num_point:
                pointcloud = np.append(pointcloud, np.zeros((self.num_point - pointcloud.shape[0], 3)), axis=0)
            pointcloud = pointcloud[:self.num_point]
            if self.data_aug:
                pointcloud = self.translate_pointcloud(pointcloud)
                np.random.shuffle(pointcloud)

            # pointcloud = self.translate_pointcloud(pointcloud)
            # np.random.shuffle(pointcloud)

        pointcloud = torch.FloatTensor(pointcloud)
        label = torch.LongTensor([label])
        pn = torch.IntTensor(pn)
        pointcloud = pointcloud.permute(1, 0)
        return pointcloud, label

    def __len__(self):
        return len(self.point_list)


'''
In the WACV paper
- Totoal 80 epochs used for training
- 400 training episodes and 600 validating episodes for each epoch
- For testing, episodes=700
- n_way=5. k_shot=1. query=15 for each classes

'''


class NShotTaskSampler(Sampler):
    def __init__(self, dataset, episode_num, k_way, n_shot, query_num):
        super().__init__(dataset)
        self.dataset = dataset
        self.episode_num = episode_num
        self.k_way = k_way
        self.n_shot = n_shot
        self.query_num = query_num
        self.label_set = self.get_label_set()
        self.data, self.label = self.dataset.point_list, self.dataset.point_label

    def get_label_set(self):
        point_label_set = np.unique(self.dataset.point_label)
        return point_label_set

    def __iter__(self):
        for _ in range(self.episode_num):
            support_list = []
            query_list = []
            picked_cls_set = np.random.choice(self.label_set, self.k_way, replace=False)

            for picked_cls in picked_cls_set:
                target_index = np.where(self.label == picked_cls)[0]
                picked_target_index = np.random.choice(target_index, self.n_shot + self.query_num, replace=False)

                support_list.append(picked_target_index[:self.n_shot])

                query_list.append(picked_target_index[self.n_shot:])

            s = np.concatenate(support_list)
            # print("the length of query_list: ", query_list)

            # print("the length is over 1")
            q = np.concatenate(query_list)

            # np.random.shuffle(s)
            '''
            For epi_index
            - it's the index used for each batch
            - the first k_way*n_shot images is the support set
            - the last k_way*query images is for the query set 
            '''
            epi_index = np.concatenate((s, q))
            yield epi_index

    def __len__(self):
        return self.episode_num


def get_sets(data_path, k_way=5, n_shot=1, query_num=15, fold=0, data_aug=False, aug=True):
    train_dataset = MiniImageNet(root=data_path, split='train', fold=fold, aug=aug, data_aug=data_aug)
    train_sampler = NShotTaskSampler(dataset=train_dataset, episode_num=400, k_way=k_way, n_shot=n_shot,
                                     query_num=query_num)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)

    val_dataset = MiniImageNet(root=data_path, split='test', fold=fold, aug=aug, data_aug=data_aug)
    val_sampler = NShotTaskSampler(dataset=val_dataset, episode_num=700, k_way=k_way, n_shot=n_shot,
                                   query_num=query_num)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler)

    return train_loader, val_loader


if __name__ == '__main__':
    # data_path='/data1/minmin/modelnet40_ply_hdf5_2048_fs'
    data_path = '/dataset/ModelNet_hdf5'
    train_lod, val_lod = get_sets(data_path=data_path)

    dataset = MiniImageNet(root=data_path)
    sampler = NShotTaskSampler(dataset=dataset, episode_num=400, k_way=5, n_shot=1, query_num=1)

    dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    for (x, pn, y) in dataloader:
        '''
        x' shape is (80,3,1024)
        y's shpae is (80,1)
        '''
        print(x.shape)

    # ========== Data Visulization ===================
    # point,label=dataset[0]
    # point=point.permute(1,0).numpy()
    # print(label)

    # pointcloud=o3d.geometry.PointCloud()
    # pointcloud.points=o3d.utility.Vector3dVector(point)
    # o3d.visualization.draw_geometries([pointcloud])
    # =================================================

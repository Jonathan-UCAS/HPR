import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import Sampler
import os
import pickle

from tqdm import tqdm
from torch.utils.data import DataLoader
np.random.seed(0)


class ModelNet40_fs(Dataset):
    def __init__(self,root,split='train',fold=0,num_point=1024,data_aug=True):
        super().__init__()
        self.root=root
        self.fold=fold
        self.split=split
        self.num_point=num_point
        self.data_aug=data_aug

        self.point_list,self.point_label=self.get_point()


    def get_point(self):
        #== will be returned later ==
        list_of_points=[]
        list_of_labels=[]
        #============================

        picked_index=np.zeros(40)
        picked_index[self.fold*10:(self.fold+1)*10]=1 # 选择某一折作为测试集,将这一部分(10类)置为1

        class_list=np.arange(40)
        if self.split=='train': #
            picked_index=(1-picked_index).astype(bool) # 1部分变为0,0部分变为1成为训练集
        else:
            picked_index=picked_index.astype(bool)

        class_list=class_list[picked_index] # 选中相应类别数
        for c in class_list:
            class_fold=os.path.join(self.root,str(c),'%d.npy' % (c)) # 第c类
            with open(class_fold, 'rb') as f:
                points = pickle.load(f)
            for i in range(len(points)):
                list_of_labels.append(c)
                list_of_points.append(points[i])

        return list_of_points,list_of_labels


    def __len__(self):
        return len(self.point_list)


    def translate_pointcloud(self,pointcloud):
        xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
        xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

        translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
        return translated_pointcloud

    def density(self,pointcloud, severity=1):
        N, C = pointcloud.shape
        c = [(1, 100), (2, 100), (3, 100), (4, 100), (5, 100)][severity - 1]
        for _ in range(c[0]):
            i = np.random.choice(pointcloud.shape[0], 1)
            picked = pointcloud[i]
            dist = np.sum((pointcloud - picked) ** 2, axis=1, keepdims=True)
            idx = np.argpartition(dist, c[1], axis=0)[:c[1]]
            idx_2 = np.random.choice(c[1], int((3 / 4) * c[1]), replace=False)
            idx = idx[idx_2]
            point_d = np.delete(pointcloud, idx.squeeze(), axis=0)
            # pointcloud[idx.squeeze()] = 0
        # print(pointcloud.shape)
        return point_d

    def background_noise(self,pointcloud, severity=1):
        N, C = pointcloud.shape
        c = [N // 45, N // 40, N // 35, N // 30, N // 20][severity - 1]
        jitter = np.random.uniform(-1, 1, (c, C))
        new_pc = np.concatenate((pointcloud, jitter), axis=0).astype('float32')
        return normalize(new_pc)


    def __getitem__(self, index):
        point = self.point_list[index][:self.num_point]  # (1024,3)
        label=self.point_label[index]

        if self.split == 'train' and self.data_aug:
            point = self.translate_pointcloud(point)
            np.random.shuffle(point)
        if self.split == 'test':
            point = self.density(point)
            # point = self.background_noise(point)
            pn = min(point.shape[0], self.num_point)
            if pn < self.num_point:
                point = np.append(point, np.zeros((self.num_point - point.shape[0], 3)), axis=0)
            point = point[:self.num_point]

        pointcloud=torch.FloatTensor(point)
        label=torch.LongTensor([label])
        pointcloud=pointcloud.permute(1,0) # 矩阵转置
        return pointcloud, label


'''
In the WACV paper
- Totoal 80 epochs used for training
- 400 training episodes and 600 validating episodes for each epoch
- For testing, episodes=700
- n_way=5. k_shot=1. query=10 for each classes

'''


class NShotTaskSampler(Sampler):
    def __init__(self,dataset,episode_num,k_way,n_shot,query_num):
        super().__init__(dataset)
        self.dataset=dataset
        self.episode_num=episode_num
        self.k_way=k_way
        self.n_shot=n_shot
        self.query_num=query_num
        self.label_set=self.get_label_set()
        self.data,self.label =self.dataset.point_list, self.dataset.point_label

    def get_label_set(self):
        point_label_set=np.unique(self.dataset.point_label)
        return point_label_set

    def __iter__(self):
        for _ in range(self.episode_num):
            support_list=[]
            query_list=[]
            picked_cls_set=np.random.choice(self.label_set,self.k_way,replace=False)

            for picked_cls in picked_cls_set:
                target_index=np.where(self.label==picked_cls)[0]
                picked_target_index=np.random.choice(target_index,self.n_shot+self.query_num,replace=False)

                support_list.append(picked_target_index[:self.n_shot])
                query_list.append(picked_target_index[self.n_shot:])

            s=np.concatenate(support_list)
            q=np.concatenate(query_list)


            '''
            For epi_index
            - it's the index used for each batch
            - the first k_way*n_shot images is the support set
            - the last k_way*query images is for the query set
            '''
            epi_index=np.concatenate((s,q))
            yield epi_index

    def __len__(self):
        return self.episode_num


def get_sets(data_path,fold=0,k_way=5,n_shot=1,query_num=15,data_aug=True):
    train_dataset=ModelNet40_fs(root=data_path,split='train',fold=fold,data_aug=data_aug)
    train_sampler=NShotTaskSampler(dataset=train_dataset,episode_num=400,k_way=k_way,n_shot=n_shot,query_num=query_num)
    train_loader=DataLoader(train_dataset,batch_sampler=train_sampler)

    val_dataset=ModelNet40_fs(root=data_path,split='test',fold=fold,data_aug=data_aug)
    val_sampler=NShotTaskSampler(dataset=val_dataset,episode_num=700,k_way=k_way,n_shot=n_shot,query_num=query_num)
    val_loader=DataLoader(val_dataset,batch_sampler=val_sampler)

    return train_loader,val_loader

def density(pointcloud, severity):
    N, C = pointcloud.shape
    c = [(1,100), (2,100), (3,100), (4,100), (5,100)][severity-1]
    for _ in range(c[0]):
        i = np.random.choice(pointcloud.shape[0],1)
        picked = pointcloud[i]
        dist = np.sum((pointcloud - picked)**2, axis=1, keepdims=True)
        idx = np.argpartition(dist, c[1], axis=0)[:c[1]]
        idx_2 = np.random.choice(c[1],int((3/4) * c[1]),replace=False)
        idx = idx[idx_2]
        pointcloud = np.delete(pointcloud, idx.squeeze(), axis=0)
        # pointcloud[idx.squeeze()] = 0
    # print(pointcloud.shape)
    return pointcloud

def cutout(pointcloud, severity):
    N, C = pointcloud.shape
    c = [(2,30), (3,30), (5,30), (7,30), (10,30)][severity-1]
    for _ in range(c[0]):
        i = np.random.choice(pointcloud.shape[0],1)
        picked = pointcloud[i]
        dist = np.sum((pointcloud - picked)**2, axis=1, keepdims=True)
        idx = np.argpartition(dist, c[1], axis=0)[:c[1]]
        # pointcloud[idx.squeeze()] = 0
        pointcloud = np.delete(pointcloud, idx.squeeze(), axis=0)
    # print(pointcloud.shape)
    return pointcloud

def rotation(pointcloud, severity):
    N, C = pointcloud.shape
    c = [2.5, 5, 7.5, 10, 15][severity - 1]
    theta = np.random.uniform(c - 2.5, c + 2.5) * np.random.choice([-1, 1]) * np.pi / 180.
    gamma = np.random.uniform(c - 2.5, c + 2.5) * np.random.choice([-1, 1]) * np.pi / 180.
    beta = np.random.uniform(c - 2.5, c + 2.5) * np.random.choice([-1, 1]) * np.pi / 180.

    matrix_1 = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
    matrix_2 = np.array([[np.cos(gamma), 0, np.sin(gamma)], [0, 1, 0], [-np.sin(gamma), 0, np.cos(gamma)]])
    matrix_3 = np.array([[np.cos(beta), -np.sin(beta), 0], [np.sin(beta), np.cos(beta), 0], [0, 0, 1]])

    new_pc = np.matmul(pointcloud, matrix_1)
    new_pc = np.matmul(new_pc, matrix_2)
    new_pc = np.matmul(new_pc, matrix_3).astype('float32')

    return normalize(new_pc)

def shear(pointcloud,severity):
    N, C = pointcloud.shape
    c = [0.05, 0.1, 0.15, 0.2, 0.25][severity-1]
    a = np.random.uniform(c-0.05,c+0.05) * np.random.choice([-1,1])
    b = np.random.uniform(c-0.05,c+0.05) * np.random.choice([-1,1])
    d = np.random.uniform(c-0.05,c+0.05) * np.random.choice([-1,1])
    e = np.random.uniform(c-0.05,c+0.05) * np.random.choice([-1,1])
    f = np.random.uniform(c-0.05,c+0.05) * np.random.choice([-1,1])
    g = np.random.uniform(c-0.05,c+0.05) * np.random.choice([-1,1])

    matrix = np.array([[1,0,b],[d,1,e],[f,0,1]])
    new_pc = np.matmul(pointcloud,matrix).astype('float32')
    return normalize(new_pc)

def scale(pointcloud,severity):
    #TODO
    N, C = pointcloud.shape
    c = [0.1, 0.2, 0.3, 0.4, 0.5][severity-1]
    a=b=d=1
    r = np.random.randint(0,3)
    t = np.random.choice([-1,1])
    if r == 0:
        a += c * t
        b += c * (-t)
    elif r == 1:
        b += c * t
        d += c * (-t)
    elif r == 2:
        a += c * t
        d += c * (-t)

    matrix = np.array([[a,0,0],[0,b,0],[0,0,d]])
    new_pc = np.matmul(pointcloud,matrix).astype('float32')
    return normalize(new_pc)

def uniform_noise(pointcloud, severity):
    #TODO
    N, C = pointcloud.shape
    c = [0.01, 0.02, 0.03, 0.04, 0.05][severity-1]
    jitter = np.random.uniform(-c,c,(N, C))
    new_pc = (pointcloud + jitter).astype('float32')
    return normalize(new_pc)

def gaussian_noise(pointcloud, severity):
    N, C = pointcloud.shape
    c = [0.01, 0.015, 0.02, 0.025, 0.03][severity-1]
    jitter = np.random.normal(size=(N, C)) * c
    new_pc = (pointcloud + jitter).astype('float32')
    new_pc = np.clip(new_pc,-1,1)
    return new_pc


'''
Add noise to the edge-length-2 cude
'''


def background_noise(pointcloud, severity):
    N, C = pointcloud.shape
    c = [N // 45, N // 40, N // 35, N // 30, N // 20][severity - 1]
    jitter = np.random.uniform(-1, 1, (c, C))
    new_pc = np.concatenate((pointcloud, jitter), axis=0).astype('float32')
    return normalize(new_pc)

'''
Upsampling
'''

def upsampling(pointcloud, severity):
    N, C = pointcloud.shape
    c = [N // 5, N // 4, N // 3, N // 2, N][severity - 1]
    index = np.random.choice(1024, c, replace=False)
    add = pointcloud[index] + np.random.uniform(-0.05, 0.05, (c, C))
    new_pc = np.concatenate((pointcloud, add), axis=0).astype('float32')
    return normalize(new_pc)
'''
Add impulse noise
'''

def impulse_noise(pointcloud, severity):
    N, C = pointcloud.shape
    c = [N // 30, N // 25, N // 20, N // 15, N // 10][severity - 1]
    index = np.random.choice(1024, c, replace=False)
    pointcloud[index] += np.random.choice([-1, 1], size=(c, C)) * 0.1
    return normalize(pointcloud)

def normalize(new_pc):
    new_pc[:,0] -= (np.max(new_pc[:,0]) + np.min(new_pc[:,0])) / 2
    new_pc[:,1] -= (np.max(new_pc[:,1]) + np.min(new_pc[:,1])) / 2
    new_pc[:,2] -= (np.max(new_pc[:,2]) + np.min(new_pc[:,2])) / 2
    leng_x, leng_y, leng_z = np.max(new_pc[:,0]) - np.min(new_pc[:,0]), np.max(new_pc[:,1]) - np.min(new_pc[:,1]), np.max(new_pc[:,2]) - np.min(new_pc[:,2])
    if leng_x >= leng_y and leng_x >= leng_z:
        ratio = 2.0 / leng_x
    elif leng_y >= leng_x and leng_y >= leng_z:
        ratio = 2.0 / leng_y
    else:
        ratio = 2.0 / leng_z
    new_pc *= ratio
    return new_pc


if __name__=='__main__':
    # root='/data1/jiajing/dataset/ModelNet40_fewshot/modelnet40_fs_crossvalidation'
    root='/dataset/ModelNet40_fs_cross_validation'
    # dataset=ModelNet40_fs(root)

    train_loader,test_loader=get_sets(data_path=root)
    for (x,y) in train_loader:
        '''
        x' shape is (80,3,1024)
        y's shpae is (80,1)
        '''
        # print(x.shape)
        # print(y.shape)
        pass



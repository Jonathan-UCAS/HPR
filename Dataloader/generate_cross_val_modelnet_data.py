import os
import numpy as np
import warnings
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D] 20000*3
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D] 1024*3
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

# class generate_data(Dataset):
#     def __init__(self, root,target, args, split='train', process_data=False):
#         self.root = root
#         self.target = target
#         self.npoints = args.num_point
#         self.process_data = process_data
#         self.uniform = args.use_uniform_sample
#         self.use_normals = args.use_normals
#         self.num_category = args.num_category
#
#         self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')
#         self.cat = [line.rstrip() for line in open(self.catfile)]
#         self.classes = dict(zip(self.cat, range(len(self.cat))))
#         self.datapath = os.path.join(self.root, '')
#
#         self.save_path = os.path.join(target, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))

def generate(path1,path2,npoints):
    class_list = np.arange(40)
    # 选中相应类别数
    for c in class_list:
        point_path_list = []
        index = 0
        class_fold = os.path.join(path1, str(c))  # 第c类 连接文件
        txts = os.listdir(class_fold)
        list_of_points = [None] * len(txts)
        for i in os.listdir(class_fold):  # 遍历第c类中的数据索引
            point_path_list.append(os.path.join(class_fold, i))
            point_set = np.loadtxt(point_path_list[index], delimiter=',').astype(np.float32)
            point_set = farthest_point_sample(point_set, npoints)
            list_of_points[index] = point_set
            index = index+1
        save_path = os.path.join(path2,str(c),'%d.npy' % (c))
        with open(save_path, 'wb') as f:
            pickle.dump(list_of_points, f)


root = '/dataset/modelnet40_normal_resampled_number'
target = '/dataset/ScanobjectNN_hdf5_PB'
# npoints = 1024

# generate(root,target,npoints)
for i in range(15):
    os.mkdir(os.path.join(target, str(i)))

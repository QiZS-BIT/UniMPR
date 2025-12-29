import os
import pickle
import numpy as np
import cv2
import math
import torch
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
from PIL import Image
from dataset.NuScenes.utils import polar_to_cartesian, cartesian_to_polar
from torchvision.transforms import transforms, autoaugment
from nuscenes.nuscenes import transform_matrix, Quaternion
import torch.nn.functional as F


class BaseDataset(Dataset):
    def __init__(self, info_root, data_root, bev_dataset_root, w=900, h=200, res=4, measure_range=80.0, trans_threshold=None):
        super(BaseDataset, self).__init__()
        self.info_root = info_root
        self.data_root = data_root
        self.bev_dataset_root = bev_dataset_root
        self.lidar_bev_root = os.path.join(self.bev_dataset_root, 'lidar')
        assert os.path.exists(self.lidar_bev_root), print('LiDAR BEV root not exists!')
        self.radar_bev_root = os.path.join(self.bev_dataset_root, 'radar')
        assert os.path.exists(self.radar_bev_root), print('Radar BEV root not exists!')

        self.cam_channels = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
        resize = [336, 336]
        self.img_transforms = transforms.Compose([
            transforms.Resize(resize, interpolation=transforms.InterpolationMode.BILINEAR),
            # autoaugment.RandAugment(num_ops=3, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.infos = self.read_info()
        self.l_bev_w = w
        self.l_bev_h = h
        self.r_bev_w = math.ceil(self.l_bev_w / res)
        self.r_bev_h = math.ceil(self.l_bev_h / res)
        self.trans_threshold = trans_threshold
        self.measure_range = measure_range

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass

    def read_info(self):
        infos = list()
        for cur_info_root in self.info_root:
            with open(cur_info_root, 'rb') as f:
                infos.extend(pickle.load(f))
        return infos

    def load_data(self, index, l_enable=True, c_enable=True, r_enable=True):
        if c_enable:
            imgs = self.load_imgs(index)
            imgs = imgs.to(dtype=torch.float32)
            intr, c_l = self.load_img_metas(index)
            lidar_points = self.load_lidar(index)
            lidar_points = lidar_points.to(dtype=torch.float32)
            f_name = None
        else:
            imgs = None
            intr = None
            c_l = None
            lidar_points = None
            f_name = None

        if r_enable:
            r_bev = self.load_r_bev(index)
            r_bev = r_bev.to(dtype=torch.float32)
        else:
            r_bev = None

        if l_enable:
            l_bev = self.load_l_bev(index)
            l_bev = l_bev.to(dtype=torch.float32)
        else:
            l_bev = None
        return imgs, intr, c_l, lidar_points, l_bev, r_bev, f_name

    def load_imgs(self, index):
        imgs = []
        for channel in self.cam_channels:
            cam_data = self.infos[index]['cam_infos'][channel]
            filename = cam_data['filename']
            img_path = os.path.join(self.data_root, filename)
            if not os.path.exists(img_path):
                raise Exception(f'FileNotFound! {img_path}')
            img = Image.open(img_path)
            img_tensor = self.img_transforms(img)
            imgs.append(img_tensor)
        imgs = torch.stack(imgs)
        return imgs

    def load_lidar(self, index):
        num_angles = 64
        num_radii = 8
        num_heights = 4
        theta = torch.linspace(0, 2 * np.pi, steps=num_angles, dtype=torch.float32)
        r = torch.linspace(10.0, 80.0, steps=num_radii, dtype=torch.float32)
        z = torch.linspace(-4.0, 12.0, steps=num_heights, dtype=torch.float32)
        theta_grid, r_grid, z_grid = torch.meshgrid(theta, r, z, indexing='ij')
        x_coords = r_grid * torch.cos(theta_grid)
        y_coords = r_grid * torch.sin(theta_grid)
        z_coords = z_grid
        point_cloud_stacked = torch.stack([x_coords, y_coords, z_coords], dim=-1)
        point_cloud = point_cloud_stacked.reshape(-1, 3)
        return point_cloud

    def load_l_bev(self, index):
        cur_info = self.infos[index]
        l_filename = os.path.basename(cur_info['lidar_infos']['LIDAR_TOP']['filename']).split('.')[0] + '.png'
        l_bev_filepath = os.path.join(self.lidar_bev_root, l_filename)
        l_bev = cv2.imread(l_bev_filepath, 0)
        l_bev = (l_bev.astype(np.float32)) / 256
        l_bev = l_bev[np.newaxis, :, :].repeat(3, 0)
        l_bev = torch.from_numpy(l_bev)
        return l_bev

    def load_r_bev(self, index):
        cur_info = self.infos[index]
        r_filename = os.path.basename(cur_info['lidar_infos']['LIDAR_TOP']['filename']).split('.')[0] + '.png'
        r_bev_filepath = os.path.join(self.radar_bev_root, r_filename)
        r_bev = cv2.imread(r_bev_filepath, 0)
        r_bev = (r_bev.astype(np.float32)) / 256
        r_bev = r_bev[np.newaxis, :, :].repeat(3, 0)
        r_bev = torch.from_numpy(r_bev)
        return r_bev

    def load_img_metas(self, index):
        # extrinsics = list()
        intrinsics = list()
        C_L = list()
        for channel in self.cam_channels:
            cam_data = self.infos[index]['cam_infos'][channel]
            # cam_extr = torch.from_numpy(cam_data['extrinsic'])
            cam_intr = torch.from_numpy(np.array(cam_data['calibrated_sensor']['camera_intrinsic'])).to(dtype=torch.float32)
            # extrinsics.append(cam_extr)
            intrinsics.append(cam_intr)

            G2E = transform_matrix(
                self.infos[index]['cam_infos'][channel]['ego_pose']['translation'],
                Quaternion(self.infos[index]['cam_infos'][channel]['ego_pose']['rotation']),
                inverse=False
            )
            E2C = transform_matrix(
                self.infos[index]['cam_infos'][channel]['calibrated_sensor']['translation'],
                Quaternion(self.infos[index]['cam_infos'][channel]['calibrated_sensor']['rotation']),
                inverse=False
            )
            G2C = np.dot(G2E, E2C)
            G2E = transform_matrix(
                self.infos[index]['lidar_infos']['LIDAR_TOP']['ego_pose']['translation'],
                Quaternion(self.infos[index]['lidar_infos']['LIDAR_TOP']['ego_pose']['rotation']),
                inverse=False
            )
            E2L = transform_matrix(
                self.infos[index]['lidar_infos']['LIDAR_TOP']['calibrated_sensor']['translation'],
                Quaternion(self.infos[index]['lidar_infos']['LIDAR_TOP']['calibrated_sensor']['rotation']),
                inverse=False
            )
            G2L = np.dot(G2E, E2L)
            C2L = np.dot(np.linalg.inv(G2C), G2L)
            C2L = torch.from_numpy(C2L).to(dtype=torch.float32)
            C_L.append(C2L)
        return intrinsics, C_L

    def rotate_polar_bev(self, l_bev, r_bev, l_enable=True, r_enable=True):
        random_rot_angle = np.random.random()
        if l_enable:
            l_bev_new = torch.zeros_like(l_bev)
            l_bev_trans = np.floor(random_rot_angle * self.l_bev_w).astype(int)
            l_bev_new[:, :, :l_bev_trans] = l_bev[:, :, :l_bev_trans]
            l_bev[:, :, 0:(self.l_bev_w - l_bev_trans)] = l_bev[:, :, l_bev_trans:]
            l_bev[:, :, (self.l_bev_w - l_bev_trans):] = l_bev_new[:, :, :l_bev_trans]
        return l_bev, r_bev, random_rot_angle


class TripletDataset(BaseDataset):
    def __init__(self, info_root, data_root, bev_dataset_root, database_root_list, query_root_list,
                 n_pos, n_neg, neg_dist_thres, pos_dist_thres, desc_dim, augmentation=True,
                 l_enable=True, c_enable=True, r_enable=True):
        super().__init__(info_root, data_root, bev_dataset_root, trans_threshold=(-pos_dist_thres / 2.0, pos_dist_thres / 2.0))
        # same elements may exist both in database and query
        self.database_root_list = database_root_list
        self.query_root_list = query_root_list
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.neg_dist_thres = neg_dist_thres
        self.pos_dist_thres = pos_dist_thres
        self.augmentation = augmentation
        self.desc_dim = desc_dim
        self.l_enable = l_enable
        self.c_enable = c_enable
        self.r_enable = r_enable

        self.len_per_seq = [0]
        self.query_len_per_seq = [0]
        self.dataset_len_per_seq = [0]
        for cur_info_file in info_root:
            with open(cur_info_file, 'rb') as f:
                self.len_per_seq.append(len(pickle.load(f)))

        assert len(self.database_root_list) != 0, print("Database root list is empty!")
        self.database_index_n_pos = np.load(self.database_root_list[0])
        self.dataset_len_per_seq.append(self.database_index_n_pos.shape[0])
        for i in range(1, len(self.database_root_list)):
            database_index_n_pos = np.load(self.database_root_list[i])
            database_index_n_pos[:, 0] = database_index_n_pos[:, 0] + self.len_per_seq[i]
            self.database_index_n_pos = np.concatenate([self.database_index_n_pos, database_index_n_pos], axis=0)
            self.dataset_len_per_seq.append(self.database_index_n_pos.shape[0])
        assert len(self.query_root_list) != 0, print("Query root list is empty!")
        self.query_index_n_pos = np.load(self.query_root_list[0])
        self.query_len_per_seq.append(self.query_index_n_pos.shape[0])
        for i in range(1, len(query_root_list)):
            query_index_n_pos = np.load(self.query_root_list[i])
            query_index_n_pos[:, 0] = query_index_n_pos[:, 0] + self.len_per_seq[i]
            self.query_index_n_pos = np.concatenate([self.query_index_n_pos, query_index_n_pos], axis=0)
            self.query_len_per_seq.append(self.query_index_n_pos.shape[0])

        self.latent_vectors = np.zeros([len(self.database_index_n_pos) + len(self.query_index_n_pos), 256])
        self.is_batch_hard_mining = False

        self.pos_index = list()
        for i in range(len(self.len_per_seq) - 1):
            knn = NearestNeighbors()
            knn.fit(self.database_index_n_pos[self.dataset_len_per_seq[i]:self.dataset_len_per_seq[i+1], 1:])
            dist, pos_index_array = knn.radius_neighbors(self.query_index_n_pos[self.query_len_per_seq[i]:self.query_len_per_seq[i+1], 1:],
                                                         radius=self.pos_dist_thres,
                                                         return_distance=True)
            self.pos_index.extend(list(pos_index_array + sum(self.dataset_len_per_seq[:i+1])))
        for i, posi in enumerate(self.pos_index):
            # pos_index = np.sort(posi[dist[i] != 0.])
            pos_index = np.sort(posi)
            self.pos_index[i] = pos_index

        self.neg_index = list()
        for i in range(len(self.len_per_seq) - 1):
            knn = NearestNeighbors()
            knn.fit(self.database_index_n_pos[self.dataset_len_per_seq[i]:self.dataset_len_per_seq[i+1], 1:])
            potential_positives = list(knn.radius_neighbors(self.query_index_n_pos[self.query_len_per_seq[i]:self.query_len_per_seq[i+1], 1:],
                                                            radius=self.neg_dist_thres,
                                                            return_distance=False))
            for pos in potential_positives:
                self.neg_index.append(np.setdiff1d(np.arange(self.dataset_len_per_seq[i+1] - self.dataset_len_per_seq[i]), pos, assume_unique=True) + sum(self.dataset_len_per_seq[:i+1]))

    def __len__(self):
        return len(self.query_index_n_pos)
        # return 38633  # train l
        # return 20054  # train c
        # return 16577  # train r
        # return 20036  # train mm

    def __getitem__(self, index):
        if index > 7934:
            while index > 7934:
                index = index - 7935

        img_list = list()
        # extr_list = list()
        c_l_list = list()
        lidar_points = list()
        l_bev_list = list()
        r_bev_list = list()
        rot_angle_list = list()
        fname_list = list()

        query_index_in_info = self.query_index_n_pos[index, 0].astype(int)
        query_img, query_intr, query_c_l, query_lidar, query_l_bev, query_r_bev, fname = (
            self.load_data(query_index_in_info, self.l_enable, self.c_enable, self.r_enable))
        fname_list.append(fname)
        if self.augmentation:
            query_l_bev, query_r_bev, rot_angle = self.rotate_polar_bev(query_l_bev, query_r_bev, self.l_enable, self.r_enable)
            # query_l_bev, query_r_bev, rot_angle = self.rotate_cartesian_bev(query_l_bev, query_r_bev, self.l_enable, self.r_enable)
            rot_angle_list.append(rot_angle)
        if self.l_enable:
            l_bev_list.append(query_l_bev)
        if self.c_enable:
            img_list.append(query_img)
            # extr_list.extend(query_extr)
            c_l_list.extend(query_c_l)
            lidar_points.append(query_lidar)
        if self.r_enable:
            r_bev_list.append(query_r_bev)

        if self.pos_index[index].shape[0] == 0:
            return None
        pos_sample = np.random.choice(self.pos_index[index], self.n_pos).astype(int)
        pos_index_in_info = self.database_index_n_pos[pos_sample, 0].astype(int)
        for i in range(len(pos_index_in_info)):
            img, _, c_l, lidar, l_bev, r_bev, fname = self.load_data(pos_index_in_info[i], self.l_enable, self.c_enable, self.r_enable)
            if self.l_enable:
                l_bev_list.append(l_bev)
            if self.c_enable:
                fname_list.append(fname)
                img_list.append(img)
                # extr_list.extend(extr)
                c_l_list.extend(c_l)
                lidar_points.append(lidar)
            if self.r_enable:
                r_bev_list.append(r_bev)

        if self.is_batch_hard_mining:
            query_in_seq_idx = 0
            for i, seq_len in enumerate(self.query_len_per_seq):
                if index < seq_len:
                    query_in_seq_idx = i - 1
            query_index = index + self.database_index_n_pos.shape[0] + sum(self.query_len_per_seq[:query_in_seq_idx])
            query_des = torch.tensor(self.latent_vectors[query_index, :])
            neg_sample = self.latent_vectors[np.array(self.neg_index[index]), :]
            neg_des = torch.tensor(neg_sample)
            dist = - torch.norm(query_des[None, :] - neg_des, dim=1) + 0.5
            result = dist.topk(self.n_neg, largest=True)
            neg_dist, neg_idx_in_sample = result.values, result.indices
            neg_idx = np.array(self.neg_index[index])[neg_idx_in_sample]
        else:
            neg_idx = np.random.choice(self.neg_index[index], self.n_neg).astype(int)
        neg_index_in_info = self.database_index_n_pos[neg_idx, 0].astype(int)
        for i in range(len(neg_index_in_info)):
            img, _, c_l, lidar, l_bev, r_bev, fname = self.load_data(neg_index_in_info[i], self.l_enable, self.c_enable, self.r_enable)
            if self.l_enable:
                l_bev_list.append(l_bev)
            if self.c_enable:
                fname_list.append(fname)
                img_list.append(img)
                # extr_list.extend(extr)
                c_l_list.extend(c_l)
                lidar_points.append(lidar)
            if self.r_enable:
                r_bev_list.append(r_bev)

        if self.l_enable:
            l_bev_tensor = torch.stack(l_bev_list, dim=0)
        else:
            l_bev_tensor = torch.tensor(1)
        if self.c_enable:
            img_tensor = torch.stack(img_list, dim=0)
            # extr_tensor = torch.stack(extr_list, dim=0)
            intr_tensor = torch.stack(query_intr, dim=0)
            c_l_tensor = torch.stack(c_l_list, dim=0)
            img_metas = dict()
            # img_metas['extr'] = extr_tensor
            img_metas['intr'] = intr_tensor
            img_metas['c_l'] = c_l_tensor
            img_metas['img_size'] = torch.tensor([1600, 900])
            img_metas['rot_angle'] = torch.tensor(rot_angle_list).to(dtype=torch.float32)
            lidar_tensor = torch.stack(lidar_points, dim=0)
        else:
            img_tensor = torch.tensor(1)
            img_metas = torch.tensor(1)
            lidar_tensor = torch.tensor(1)
        if self.r_enable:
            r_bev_tensor = torch.stack(r_bev_list, dim=0)
        else:
            r_bev_tensor = torch.tensor(1)

        res_dict = dict({'image': img_tensor, 'img_metas': img_metas, 'lidar': lidar_tensor,
                         'lidar_bev': l_bev_tensor, 'radar_bev': r_bev_tensor})
        return res_dict

    def update_latent_vectors(self, vecs_filepath):
        latent_vectors = pickle.load(open(vecs_filepath, 'rb'))
        latent_vectors_full = np.zeros([len(self.database_index_n_pos) + len(self.query_index_n_pos), self.desc_dim])
        for i in range(len(latent_vectors)):
            latent_vectors_full[i, :] = latent_vectors[i]
        self.latent_vectors = latent_vectors_full
        self.is_batch_hard_mining = True


class QueryDataset(BaseDataset):
    def __init__(self, info_root, data_root, bev_dataset_root, database_root_list, query_root_list, non_triv_pos_dist_thres,
                 l_enable=True, c_enable=True, r_enable=True):
        # database and query are separated
        super().__init__(info_root, data_root, bev_dataset_root)
        self.database_root_list = database_root_list
        self.query_root_list = query_root_list
        self.positives = None
        self.non_triv_pos_dist_thres = non_triv_pos_dist_thres
        self.l_enable = l_enable
        self.c_enable = c_enable
        self.r_enable = r_enable

        self.len_per_seq = [0]
        self.query_len_per_seq = [0]
        self.dataset_len_per_seq = [0]
        for cur_info_file in info_root:
            with open(cur_info_file, 'rb') as f:
                self.len_per_seq.append(len(pickle.load(f)))

        # assert len(self.database_root_list) != 0, print("Database root list is empty!")
        if len(self.database_root_list) != 0:
            self.database_index_n_pos = np.load(self.database_root_list[0])
            self.dataset_len_per_seq.append(self.database_index_n_pos.shape[0])
            for i in range(1, len(self.database_root_list)):
                database_index_n_pos = np.load(self.database_root_list[i])
                database_index_n_pos[:, 0] = database_index_n_pos[:, 0] + self.len_per_seq[i]
                self.database_index_n_pos = np.concatenate([self.database_index_n_pos, database_index_n_pos], axis=0)
                self.dataset_len_per_seq.append(self.database_index_n_pos.shape[0])
        # assert len(self.query_root_list) != 0, print("Query root list is empty!")
        if len(self.query_root_list) != 0:
            self.query_index_n_pos = np.load(self.query_root_list[0])
            self.query_len_per_seq.append(self.query_index_n_pos.shape[0])
            for i in range(1, len(query_root_list)):
                query_index_n_pos = np.load(self.query_root_list[i])
                query_index_n_pos[:, 0] = query_index_n_pos[:, 0] + self.len_per_seq[i]
                self.query_index_n_pos = np.concatenate([self.query_index_n_pos, query_index_n_pos], axis=0)
                self.query_len_per_seq.append(self.query_index_n_pos.shape[0])

        self.dataset_index_n_pos = np.concatenate([self.database_index_n_pos, self.query_index_n_pos], axis=0)
        self.num_db = self.database_index_n_pos.shape[0]
        self.num_query = self.query_index_n_pos.shape[0]

    def __len__(self):
        return len(self.dataset_index_n_pos)

    def __getitem__(self, index):
        img_list = list()
        # extr_list = list()
        intr_list = list()
        c_l_list = list()
        lidar_points = list()
        l_bev_list = list()
        r_bev_list = list()
        index_in_info = self.dataset_index_n_pos[index, 0].astype(int)
        img, intr, c_l, lidar, l_bev, r_bev, _ = self.load_data(index_in_info, self.l_enable, self.c_enable, self.r_enable)
        if self.l_enable:
            l_bev_list.append(l_bev)
        if self.c_enable:
            img_list.append(img)
            # extr_list.extend(extr)
            intr_list.extend(intr)
            c_l_list.extend(c_l)
            lidar_points.append(lidar)
        if self.r_enable:
            r_bev_list.append(r_bev)

        if self.l_enable:
            l_bev_tensor = torch.stack(l_bev_list, dim=0)
        else:
            l_bev_tensor = torch.tensor(1)
        if self.c_enable:
            img_tensor = torch.stack(img_list, dim=0)
            # extr_tensor = torch.stack(extr_list, dim=0)
            intr_tensor = torch.stack(intr_list, dim=0)
            c_l_tensor = torch.stack(c_l_list, dim=0)
            img_metas = dict()
            # img_metas['extr'] = extr_tensor
            img_metas['intr'] = intr_tensor
            img_metas['c_l'] = c_l_tensor
            img_metas['img_size'] = torch.tensor([1600, 900])
            lidar_tensor = torch.stack(lidar_points, dim=0)
        else:
            img_tensor = torch.tensor(1)
            img_metas = torch.tensor(1)
            lidar_tensor = torch.tensor(1)
        if self.r_enable:
            r_bev_tensor = torch.stack(r_bev_list, dim=0)
        else:
            r_bev_tensor = torch.tensor(1)

        res_dict = dict({'image': img_tensor, 'img_metas': img_metas, 'lidar': lidar_tensor,
                         'lidar_bev': l_bev_tensor, 'radar_bev': r_bev_tensor})
        return res_dict

    def get_positives(self):
        # positives for evaluation are those within trivial threshold range
        # fit NN to find them, search by radius
        if self.positives is None:
            knn = NearestNeighbors()
            dataset = np.ascontiguousarray(self.dataset_index_n_pos[:self.num_db, 1:])
            knn.fit(dataset)
            self.positives = list(knn.radius_neighbors(self.dataset_index_n_pos[self.num_db:, 1:],
                                                       radius=self.non_triv_pos_dist_thres,
                                                       return_distance=False))
        return self.positives

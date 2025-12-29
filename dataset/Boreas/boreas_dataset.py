import os
import pickle
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
from torchvision.transforms import transforms, autoaugment
from PIL import Image
from dataset.Boreas.utils import filter_by_yaw


class BaseDataset(Dataset):
    def __init__(self, info_root, bev_dataset_root):
        super(BaseDataset, self).__init__()
        self.info_root = info_root
        self.bev_dataset_root = bev_dataset_root
        self.lidar_bev_root = os.path.join(self.bev_dataset_root, 'lidar')
        self.radar_bev_root = os.path.join(self.bev_dataset_root, 'radar')

        self.cam_channels = ['Cam']
        resize = [336, 336]
        self.img_transforms = transforms.Compose([
            transforms.Resize(resize, interpolation=transforms.InterpolationMode.BILINEAR),
            # autoaugment.RandAugment(num_ops=3, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.infos = self.read_info()

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
        else:
            imgs = None
            intr = None
            c_l = None
            lidar_points = None

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
        return imgs, intr, c_l, lidar_points, l_bev, r_bev

    def load_imgs(self, index):
        imgs = []
        for channel in self.cam_channels:
            cam_data = self.infos[index]['cam_infos'][channel]
            img_path = cam_data['filename']
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
        z = torch.linspace(-14.0, 14.0, steps=num_heights, dtype=torch.float32)
        theta_grid, r_grid, z_grid = torch.meshgrid(theta, r, z, indexing='ij')
        x_coords = r_grid * torch.cos(theta_grid)
        y_coords = r_grid * torch.sin(theta_grid)
        z_coords = z_grid
        point_cloud_stacked = torch.stack([x_coords, y_coords, z_coords], dim=-1)
        point_cloud = point_cloud_stacked.reshape(-1, 3)
        return point_cloud

    # Downsample the LiDAR point cloud to 2,048 points and using them as query
    # can further improve the performance of the camera branch
    # def load_lidar(self, index):
    #     lidar_data = self.infos[index]['lidar_infos']['LIDAR_TOP']
    #     f_name = os.path.basename(lidar_data['filename']).split('.')[0] + '.npy'
    #     self.downsampled_lidar_root = ''
    #     f_path = os.path.join(self.downsampled_lidar_root, f_name)
    #     pc = np.load(f_path, allow_pickle=True)
    #     pc = torch.from_numpy(pc)
    #     return pc

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
        r_filename = os.path.basename(cur_info['radar_infos']['RADAR_TOP']['filename']).split('.')[0] + '.png'
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
            if channel == 'Cam':
                cam_intr = torch.from_numpy(np.array([[1436.89437780124, 0, 1218.77863491507], [0, 1442.58498490709, 1045.77362896196], [0, 0, 1]])).to(dtype=torch.float32)
            else:
                cam_intr = None
            intrinsics.append(cam_intr)

            if channel == 'Cam':
                C2L = np.array([[0.729304194569723, -0.684189392429244, -0.000516788457270013, -0.0231487194000858],
                                [-0.0128793477606882, -0.0129733964677743, -0.999832892730255, -0.229363448544232],
                                [0.68406835490634, 0.729188978435215, -0.018273465581034, -0.635346228374209],
                                [0, 0, 0, 1]])
                theta_degrees = 133.026723
                theta_radians = np.radians(theta_degrees)
                transformation_matrix = np.array([[np.cos(theta_radians), -np.sin(theta_radians), 0., 0.],
                                                  [np.sin(theta_radians), np.cos(theta_radians), 0, 0.],
                                                  [0., 0., 1., 0.],
                                                  [0., 0., 0., 1.]])
                transformation_matrix = np.linalg.inv(transformation_matrix)
                C2L = np.dot(C2L, transformation_matrix)
            else:
                C2L = None
            C2L = torch.from_numpy(C2L).to(dtype=torch.float32)
            C_L.append(C2L)
        return intrinsics, C_L


class QueryDataset(BaseDataset):
    def __init__(self, info_root, bev_dataset_root, database_root_list, query_root_list, non_triv_pos_dist_thres, pos_rot_thres,
                 l_enable=True, c_enable=True, r_enable=True):
        # database and query are separated
        super().__init__(info_root, bev_dataset_root)
        self.database_root_list = database_root_list
        self.query_root_list = query_root_list
        self.positives = None
        self.non_triv_pos_dist_thres = non_triv_pos_dist_thres
        self.pos_rot_thres = pos_rot_thres
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
        intr_list = list()
        c_l_list = list()
        lidar_points = list()
        l_bev_list = list()
        r_bev_list = list()
        index_in_info = self.dataset_index_n_pos[index, 0].astype(int)

        img, intr, c_l, lidar, l_bev, r_bev = self.load_data(index_in_info, self.l_enable, self.c_enable, self.r_enable)
        if self.c_enable:
            img_list.append(img)
            intr_list.extend(intr)
            c_l_list.extend(c_l)
            lidar_points.append(lidar)
        if self.l_enable:
            l_bev_list.append(l_bev)
        if self.r_enable:
            r_bev_list.append(r_bev)

        if self.l_enable:
            l_bev_tensor = torch.stack(l_bev_list, dim=0)
        else:
            l_bev_tensor = torch.tensor(1)
        if self.c_enable:
            c_usable_flag = True
            if c_usable_flag:
                img_tensor = torch.stack(img_list, dim=0)
                intr_tensor = torch.stack(intr_list, dim=0)
                c_l_tensor = torch.stack(c_l_list, dim=0)
                img_metas = dict()
                img_metas['intr'] = intr_tensor
                img_metas['c_l'] = c_l_tensor
                img_metas['img_size'] = torch.tensor([2448, 2048])
                lidar_tensor = torch.stack(lidar_points, dim=0)
            else:
                img_tensor = torch.stack(img_list, dim=0)
                intr_tensor = torch.stack(intr_list, dim=0)
                c_l_tensor = torch.stack(c_l_list, dim=0)
                img_metas = dict()
                img_metas['intr'] = torch.zeros_like(intr_tensor)
                img_metas['c_l'] = torch.zeros_like(c_l_tensor)
                img_metas['img_size'] = torch.tensor([0, 0])
                lidar_tensor = torch.zeros_like(torch.stack(lidar_points, dim=0))
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
            dataset = np.ascontiguousarray(self.dataset_index_n_pos[:self.num_db, 1:3])
            knn.fit(dataset)
            self.positives = list(knn.radius_neighbors(self.dataset_index_n_pos[self.num_db:, 1:3],
                                                       radius=self.non_triv_pos_dist_thres,
                                                       return_distance=False))
            if self.c_enable is True and self.l_enable is False and self.r_enable is False:
                filtered_pos_index_array = filter_by_yaw(
                    self.positives,
                    self.dataset_index_n_pos[self.num_db:],
                    self.dataset_index_n_pos[:self.num_db],
                    angle_threshold=self.pos_rot_thres)
                self.positives = filtered_pos_index_array
        return self.positives

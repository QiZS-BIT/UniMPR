import os
import pickle
import cv2
import open3d as o3d
import numpy as np
from dataset.Boreas.utils import load_radar_polar_pool, load_pc
from tqdm import tqdm


def lidar_polar_bev_projection(pcl_input, measure_range, resolution, w=900, h=200):
    pcl_input_pc = o3d.geometry.PointCloud()
    pcl_input_pc.points = o3d.utility.Vector3dVector(pcl_input)
    pcl_input_pc = pcl_input_pc.voxel_down_sample(voxel_size=resolution)
    pcl_input_np = np.asarray(pcl_input_pc.points)

    mat_global_image = np.zeros((h, w), dtype=np.uint8)

    yaw = -np.arctan2(pcl_input_np[:, 1], pcl_input_np[:, 0])
    scan_r = np.linalg.norm(pcl_input_np[:, :2], axis=1)

    w_ind = np.floor((0.5 * (yaw / np.pi + 1.0)) * w).astype(int)
    h_ind = np.floor(scan_r / measure_range * h).astype(int)

    valid_mask = (w_ind >= 0) & (w_ind < w) & (h_ind >= 0) & (h_ind < h)

    valid_w_ind = w_ind[valid_mask]
    valid_h_ind = h_ind[valid_mask]

    np.add.at(mat_global_image, (valid_h_ind, valid_w_ind), 1)
    mat_global_image = np.clip(mat_global_image, 0, 10)

    mat_global_image = mat_global_image * 10

    mat_global_image[np.where(mat_global_image > 255)] = 255
    mat_global_image = mat_global_image / np.max(mat_global_image) * 255

    return mat_global_image


def gen_bev(bev_fileroot, info_filepath, bs_whole):
    if not os.path.exists(bev_fileroot):
        os.makedirs(bev_fileroot)
    lidar_bev_root = os.path.join(bev_fileroot, 'lidar')
    if not os.path.exists(lidar_bev_root):
        os.makedirs(lidar_bev_root)
    radar_bev_root = os.path.join(bev_fileroot, 'radar')
    if not os.path.exists(radar_bev_root):
        os.makedirs(radar_bev_root)

    with open(info_filepath, 'rb') as f:
        infos = pickle.load(f)

    for i in tqdm(range(0, bs_whole.shape[0])):
        cur_ind = bs_whole[i][0]
        cur_info = infos[int(cur_ind)]

        radar_filename = cur_info['radar_infos']['RADAR_TOP']['filename']
        r_bev_img = load_radar_polar_pool(radar_filename, 225, 50)
        lidar_filename = cur_info['lidar_infos']['LIDAR_TOP']['filename']
        lidar_pcd = load_pc(lidar_filename)

        l_bev_img = lidar_polar_bev_projection(lidar_pcd, 80.0, 0.4, 900, 200)

        filename_l = os.path.basename(cur_info['lidar_infos']['LIDAR_TOP']['filename']).split('.')[0] + '.png'
        filename_r_img = os.path.basename(cur_info['radar_infos']['RADAR_TOP']['filename']).split('.')[0] + '.png'
        cv2.imwrite(os.path.join(radar_bev_root, filename_r_img), r_bev_img)
        cv2.imwrite(os.path.join(lidar_bev_root, filename_l), l_bev_img)


if __name__ == '__main__':
    dataset_root = '/home/octane17/UniMPR/data/boreas'

    # ----------------------------------------------------------------
    bev_root = os.path.join(dataset_root, 'bev', 'mm_bev')
    infos_path = os.path.join(dataset_root, 'info', 'boreas_infos-2021-01-26-11-22.pkl')
    db_ind_root = os.path.join(dataset_root, 'index', 'boreas_infos-2021-01-26-11-22_db.npy')
    test_query_ind_root = os.path.join(dataset_root, 'index', 'boreas_infos-2021-01-26-11-22_test_query.npy')
    ind_db = np.load(db_ind_root)
    ind_test_query = np.load(test_query_ind_root)
    ind_whole = np.concatenate([ind_db, ind_test_query], axis=0)
    gen_bev(bev_root, infos_path, ind_whole)

    # ----------------------------------------------------------------
    bev_root = os.path.join(dataset_root, 'bev', 'mm_bev')
    infos_path = os.path.join(dataset_root, 'info', 'boreas_infos-2021-04-29-15-55.pkl')
    db_ind_root = os.path.join(dataset_root, 'index', 'boreas_infos-2021-04-29-15-55_db.npy')
    test_query_ind_root = os.path.join(dataset_root, 'index', 'boreas_infos-2021-04-29-15-55_test_query.npy')
    ind_db = np.load(db_ind_root)
    ind_test_query = np.load(test_query_ind_root)
    ind_whole = np.concatenate([ind_db, ind_test_query], axis=0)
    gen_bev(bev_root, infos_path, ind_whole)

    # ----------------------------------------------------------------
    bev_root = os.path.join(dataset_root, 'bev', 'mm_bev')
    infos_path = os.path.join(dataset_root, 'info', 'boreas_infos-2021-09-14-20-00.pkl')
    db_ind_root = os.path.join(dataset_root, 'index', 'boreas_infos-2021-09-14-20-00_db.npy')
    test_query_ind_root = os.path.join(dataset_root, 'index', 'boreas_infos-2021-09-14-20-00_test_query.npy')
    ind_db = np.load(db_ind_root)
    ind_test_query = np.load(test_query_ind_root)
    ind_whole = np.concatenate([ind_db, ind_test_query], axis=0)
    gen_bev(bev_root, infos_path, ind_whole)

    # ----------------------------------------------------------------
    bev_root = os.path.join(dataset_root, 'bev', 'mm_bev')
    infos_path = os.path.join(dataset_root, 'info', 'boreas_infos-2021-11-16-14-10.pkl')
    db_ind_root = os.path.join(dataset_root, 'index', 'boreas_infos-2021-11-16-14-10_db.npy')
    test_query_ind_root = os.path.join(dataset_root, 'index', 'boreas_infos-2021-11-16-14-10_test_query.npy')
    ind_db = np.load(db_ind_root)
    ind_test_query = np.load(test_query_ind_root)
    ind_whole = np.concatenate([ind_db, ind_test_query], axis=0)
    gen_bev(bev_root, infos_path, ind_whole)

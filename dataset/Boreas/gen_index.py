import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
import random
import pickle


def process_infos(infos_filepath, test_seq_name, dis_th_db=1.0, angle_th_db=1, pos_th=9.0, angle_th=60):
    with open(infos_filepath, 'rb') as f:
        infos = pickle.load(f)

    pos_whole = []
    yaw_whole = []
    angle_th_db = angle_th_db * np.pi / 180
    angle_th = angle_th * np.pi / 180
    sequence_names = []

    for i, info in enumerate(infos):
        pos = info['lidar_infos']['LIDAR_TOP']['ego_pose']
        pos_whole.append(pos)
        yaw = info['cam_infos']['Cam']['yaw']
        yaw_whole.append(yaw)
        sequence_name = info['sequence_name']
        sequence_names.append(sequence_name)

    pos_whole = np.array(pos_whole, dtype=np.float32)
    yaw_whole = np.array(yaw_whole, dtype=np.float32).reshape(-1, 1)
    sequence_names = np.array(sequence_names).reshape(-1, 1)
    print('total frames: ', pos_whole.shape[0])

    fi_db_train, _ = np.where(sequence_names != test_seq_name)
    fi_val_test, _ = np.where(sequence_names == test_seq_name)

    pos_whole = np.concatenate(
        (np.arange(len(pos_whole), dtype=np.int32).reshape(-1, 1), np.array(pos_whole)),
        axis=1).astype(np.float32)
    pos_db_train = pos_whole[fi_db_train]
    pos_db = pos_db_train[0, :].reshape(1, -1)  # add the first frame
    yaw_db_train = yaw_whole[fi_db_train]
    yaw_db = yaw_db_train[0].reshape(1, -1)
    for i in range(1, pos_db_train.shape[0]):
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(pos_db[:, 1:3])
        dis, index = knn.kneighbors(pos_db_train[i, 1:3].reshape(1, -1), 1, return_distance=True)
        ref_yaw = yaw_db[index.flatten()[0], 0]  # 参考点的航向角
        current_yaw = yaw_db_train[i, 0]
        angle_diff = np.abs(current_yaw - ref_yaw)
        min_yaw_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
        if dis > dis_th_db or min_yaw_diff > angle_th_db:
            pos_db = np.concatenate((pos_db, pos_db_train[i, :].reshape(1, -1)), axis=0)
            yaw_db = np.concatenate((yaw_db, yaw_db_train[i].reshape(1, -1)), axis=0)
    print("database frames: ", pos_db.shape[0])

    pos_val_test = pos_whole[fi_val_test]
    pos_val_test_dsp = pos_val_test[0, :].reshape(1, -1)  # add the first frame
    yaw_val_test = yaw_whole[fi_val_test]
    yaw_val_test_dsp = yaw_val_test[0].reshape(1, -1)
    for i in range(1, pos_val_test.shape[0]):
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(pos_val_test_dsp[:, 1:3])
        dis, index = knn.kneighbors(pos_val_test[i, 1:3].reshape(1, -1), 1, return_distance=True)
        ref_yaw = yaw_val_test_dsp[index.flatten()[0], 0]  # 参考点的航向角
        current_yaw = yaw_val_test[i, 0]
        angle_diff = np.abs(current_yaw - ref_yaw)
        min_yaw_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
        if dis > dis_th_db or min_yaw_diff > angle_th_db:
            pos_val_test_dsp = np.concatenate((pos_val_test_dsp, pos_val_test[i, :].reshape(1, -1)), axis=0)
            yaw_val_test_dsp = np.concatenate((yaw_val_test_dsp, yaw_val_test[i].reshape(1, -1)), axis=0)
    pos_val_test = pos_val_test_dsp
    yaw_val_test = yaw_val_test_dsp

    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(pos_db[:, 1:3])
    pos_val_test_new = list()
    yaw_val_test_new = list()
    for i in range(len(pos_val_test)):
        # dis, index = knn.kneighbors(pos_val_test[i, 1:3].reshape(1, -1), 1, return_distance=True)
        dis, index = knn.radius_neighbors(pos_val_test[i, 1:3].reshape(1, -1), radius=pos_th, return_distance=True, sort_results=True)
        dis = dis.flatten()[0]
        if len(dis) == 0:
            continue
        index = index.flatten()[0]
        current_yaw = yaw_val_test[i]
        angle_diffs = np.abs(current_yaw - yaw_db[index].flatten())
        min_angles = np.minimum(angle_diffs, 2 * np.pi - angle_diffs)
        min_angle = np.min(min_angles)
        if dis[0] < pos_th:
            pos_val_test_new.append(pos_val_test[i, :])
            if min_angle < angle_th:
                yaw_val_test_new.append(yaw_val_test[i])
            else:
                yaw_val_test_new.append(10000 * np.ones_like(yaw_val_test[i]))
    pos_val_test = np.array(pos_val_test_new)
    yaw_val_test = np.array(yaw_val_test_new)
    print("test query frames: ", pos_val_test.shape[0])

    pos_whole = np.concatenate((pos_whole, yaw_whole), axis=-1)
    pos_db = np.concatenate((pos_db, yaw_db), axis=-1)
    pos_val_test = np.concatenate((pos_val_test, yaw_val_test), axis=-1)

    return pos_whole, pos_db, pos_val_test


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    dataroot = '/home/octane17/UniMPR/data/boreas'

    infos_path = os.path.join(dataroot, 'info', 'boreas_infos-2021-01-26-11-22.pkl')
    test_sequence_name = 'boreas-2021-01-26-11-22'
    pos_whole, pos_db, pos_test_query = process_infos(infos_path, test_sequence_name, 0.2, pos_th=9.0)
    np.save(os.path.join(dataroot, 'index', 'boreas_infos-2021-01-26-11-22_whole.npy'), pos_whole)
    np.save(os.path.join(dataroot, 'index', 'boreas_infos-2021-01-26-11-22_db.npy'), pos_db)
    np.save(os.path.join(dataroot, 'index', 'boreas_infos-2021-01-26-11-22_test_query.npy'), pos_test_query)

    infos_path = os.path.join(dataroot, 'info', 'boreas_infos-2021-04-29-15-55.pkl')
    test_sequence_name = 'boreas-2021-04-29-15-55'
    pos_whole, pos_db, pos_test_query = process_infos(infos_path, test_sequence_name, 0.2, pos_th=9.0)
    np.save(os.path.join(dataroot, 'index', 'boreas_infos-2021-04-29-15-55_whole.npy'), pos_whole)
    np.save(os.path.join(dataroot, 'index', 'boreas_infos-2021-04-29-15-55_db.npy'), pos_db)
    np.save(os.path.join(dataroot, 'index', 'boreas_infos-2021-04-29-15-55_test_query.npy'), pos_test_query)

    infos_path = os.path.join(dataroot, 'info', 'boreas_infos-2021-09-14-20-00.pkl')
    test_sequence_name = 'boreas-2021-09-14-20-00'
    pos_whole, pos_db, pos_test_query = process_infos(infos_path, test_sequence_name, 0.2, pos_th=9.0)
    np.save(os.path.join(dataroot, 'index', 'boreas_infos-2021-09-14-20-00_whole.npy'), pos_whole)
    np.save(os.path.join(dataroot, 'index', 'boreas_infos-2021-09-14-20-00_db.npy'), pos_db)
    np.save(os.path.join(dataroot, 'index', 'boreas_infos-2021-09-14-20-00_test_query.npy'), pos_test_query)

    infos_path = os.path.join(dataroot, 'info', 'boreas_infos-2021-11-16-14-10.pkl')
    test_sequence_name = 'boreas-2021-11-16-14-10'
    pos_whole, pos_db, pos_test_query = process_infos(infos_path, test_sequence_name, 0.2, pos_th=9.0)
    np.save(os.path.join(dataroot, 'index', 'boreas_infos-2021-11-16-14-10_whole.npy'), pos_whole)
    np.save(os.path.join(dataroot, 'index', 'boreas_infos-2021-11-16-14-10_db.npy'), pos_db)
    np.save(os.path.join(dataroot, 'index', 'boreas_infos-2021-11-16-14-10_test_query.npy'), pos_test_query)

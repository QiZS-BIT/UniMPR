import os


class ModelParams:
    def __init__(self):
        self.nusc_root = "/media/octane17/T7ShieldNus/NuScenes"
        self.base_root = "/home/octane17/UniMPR/data/nusc"

        # BS
        self.info_root = [os.path.join(self.base_root, "info", "nuscenes_infos-bs.pkl")]
        self.bev_dataset_root = os.path.join(self.base_root, "bev", "mm_bev")
        self.database_index_list = [os.path.join(self.base_root, "index", "bs_db.npy")]
        self.train_query_index_list = [os.path.join(self.base_root, "index", "bs_train_query.npy")]
        self.val_query_index_list = [os.path.join(self.base_root, "index", "bs_test_query.npy")]
        self.test_query_index_list = [os.path.join(self.base_root, "index", "bs_test_query.npy")]

        # # SON
        # self.info_root = [os.path.join(self.base_root, "info", "nuscenes_infos-son.pkl")]
        # self.bev_dataset_root = os.path.join(self.base_root, "bev", "mm_bev")
        # self.database_index_list = [os.path.join(self.base_root, "index", "son_db.npy")]
        # self.val_query_index_list = [os.path.join(self.base_root, "index", "son_test_query.npy")]
        # self.test_query_index_list = [os.path.join(self.base_root, "index", "son_test_query.npy")]

        # # SQ
        # self.info_root = [os.path.join(self.base_root, "info", "nuscenes_infos-sq.pkl")]
        # self.bev_dataset_root = os.path.join(self.base_root, "bev", "mm_bev")
        # self.database_index_list = [os.path.join(self.base_root, "index", "sq_db.npy")]
        # self.val_query_index_list = [os.path.join(self.base_root, "index", "sq_test_query.npy")]
        # self.test_query_index_list = [os.path.join(self.base_root, "index", "sq_test_query.npy")]

        self.pos_dist_threshold = 9
        self.neg_dist_threshold = 12
        self.gth_dist_threshold = 9

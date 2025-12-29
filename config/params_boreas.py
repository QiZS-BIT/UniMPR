import os


class ModelParams:
    def __init__(self):
        self.oxford_root = "/media/octane17/T7 Shield/Boreas"
        self.base_root = "/home/octane17/UniMPR/data/boreas"

        # test
        self.info_root = [os.path.join(self.base_root, "info", "boreas_infos-2021-04-29-15-55.pkl")]
        self.bev_dataset_root = os.path.join(self.base_root, "bev", "mm_bev")
        self.database_index_list = [os.path.join(self.base_root, "index", "boreas_infos-2021-04-29-15-55_db.npy")]
        self.test_query_index_list = [os.path.join(self.base_root, "index", "boreas_infos-2021-04-29-15-55_test_query.npy")]

        # self.info_root = [os.path.join(self.base_root, "info", "boreas_infos-2021-09-14-20-00.pkl")]
        # self.bev_dataset_root = os.path.join(self.base_root, "bev", "mm_bev")
        # self.database_index_list = [os.path.join(self.base_root, "index", "boreas_infos-2021-09-14-20-00_db.npy")]
        # self.test_query_index_list = [os.path.join(self.base_root, "index", "boreas_infos-2021-09-14-20-00_test_query.npy")]

        # self.info_root = [os.path.join(self.base_root, "info", "boreas_infos-2021-11-16-14-10.pkl")]
        # self.bev_dataset_root = os.path.join(self.base_root, "bev", "mm_bev")
        # self.database_index_list = [os.path.join(self.base_root, "index", "boreas_infos-2021-11-16-14-10_db.npy")]
        # self.test_query_index_list = [os.path.join(self.base_root, "index", "boreas_infos-2021-11-16-14-10_test_query.npy")]

        # self.info_root = [os.path.join(self.base_root, "info", "boreas_infos-2021-01-26-11-22.pkl")]
        # self.bev_dataset_root = os.path.join(self.base_root, "bev", "mm_bev")
        # self.database_index_list = [os.path.join(self.base_root, "index", "boreas_infos-2021-01-26-11-22_db.npy")]
        # self.test_query_index_list = [os.path.join(self.base_root, "index", "boreas_infos-2021-01-26-11-22_test_query.npy")]

        self.pos_dist_threshold = 9
        self.neg_dist_threshold = 12
        self.gth_dist_threshold = 9
        self.pos_rot_threshold = 60  # only serves a purpose when evaluating the camera branch individually

import os
from config.params_nusc import ModelParams as NuscParams
from config.params_boreas import ModelParams as BoreasParams


class ModelParamsFull:
    def __init__(self):
        self.l_enable = True
        self.c_enable = True
        self.r_enable = True

        self.nusc_param = NuscParams()
        self.boreas_param = BoreasParams()

        self.checkpoint_path = "/home/octane17/UniMPR/weights/pretrained.pth.tar"

        self.resume_checkpoint = True
        self.num_test_workers = 8
        self.test_batch_size = 1

        self.output_dim = 1024

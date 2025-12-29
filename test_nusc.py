#!/usr/bin/env python3

import os
import pickle
import sys
import random
import numpy as np
from tqdm import tqdm
import faiss
import torch
from config.params_full import ModelParamsFull
from modules.LCRFusionGCNExp import LCRFusionGCNExp
from dataset.NuScenes.nuscenes_dataloader import make_dataloader

sys.maxsize = 3000
np.set_printoptions(threshold=sys.maxsize)


class Tester:
    def __init__(self, params: ModelParamsFull):
        super(Tester, self).__init__()
        self.params = params
        self.resume = self.params.resume_checkpoint

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LCRFusionGCNExp.create()
        n_params = sum([parameter.nelement() for parameter in self.model.parameters()])
        print(f'Number of model parameters: %d' % n_params)
        self.model.to(self.device)
        self.test_query_loader, self.test_query_dataset = make_dataloader('test', params)

    def do_test(self):
        print("\nEvaluating Process-" + "-------->")

        if self.resume:
            if self.params.checkpoint_path:
                resume_filename = self.params.checkpoint_path
                checkpoint = torch.load(resume_filename)
                self.model.load_state_dict(checkpoint['state_dict'])
        else:
            print("\nZero-shot Test")

        descriptor_query = []
        descriptor_database = []
        with torch.no_grad():
            self.model.eval()
            for batch_idx, current_batch in enumerate(tqdm(self.test_query_loader)):
                image_cuda = current_batch['image'].to(self.device)
                lidar_cuda = current_batch['lidar'].to(self.device)
                img_meta_cuda = {key: value.to(self.device) for key, value in current_batch['img_metas'].items()}
                l_bev_cuda = current_batch['lidar_bev'].to(self.device)
                r_bev_cuda = current_batch['radar_bev'].to(self.device)
                _, _, _, descriptor_concat = self.model(image_cuda, lidar_cuda, img_meta_cuda, l_bev_cuda, r_bev_cuda)
                for des_idx in range(descriptor_concat.shape[0]):
                    if (batch_idx * self.params.test_batch_size + des_idx) < self.test_query_dataset.num_db:
                        descriptor_database.append(descriptor_concat[des_idx, :].cpu().detach().numpy())
                    else:
                        descriptor_query.append(descriptor_concat[des_idx, :].cpu().detach().numpy())
                torch.cuda.empty_cache()

        total_query_pcd_num = self.test_query_dataset.num_query
        descriptor_query = np.array(descriptor_query).astype('float32')
        descriptor_database = np.array(descriptor_database).astype('float32')
        faiss_index = faiss.IndexFlatL2(self.params.output_dim)
        faiss_index.add(descriptor_database)
        dis, pred_ind = faiss_index.search(descriptor_query, 50)

        no_gt_num = 0
        gt = self.test_query_dataset.get_positives()
        top_n = [1, 5, 10, 20, int(total_query_pcd_num * 0.01)]
        successful_recall_num = np.zeros(len(top_n))
        for query_ind in range(total_query_pcd_num):
            if len(gt[query_ind]) == 0:
                no_gt_num += 1
                continue
            for ind, n in enumerate(top_n):
                if np.any(np.in1d(pred_ind[query_ind, :n], np.array(gt[query_ind]))):
                    successful_recall_num[ind:] += 1
                    break
        total_query_pcd_num -= no_gt_num

        recall_rate = successful_recall_num / float(total_query_pcd_num) * 100.0
        print(f'\nRecall@1: %.2f' % float(recall_rate[0]))
        print(f'Recall@5: %.2f' % float(recall_rate[1]))
        print(f'Recall@10: %.2f' % float(recall_rate[2]))
        print(f'Recall@20: %.2f' % float(recall_rate[3]))
        print(f'Recall@1percent: %.2f\n' % float(recall_rate[4]))

        faiss_index = faiss.IndexFlatL2(self.params.output_dim)
        faiss_index.add(descriptor_database)
        dists, preds = faiss_index.search(descriptor_query, len(descriptor_database))  # the results are sorted
        dists_max = dists[:, 0].max()
        dists_min = dists[:, 0].min()
        if dists_min - 0.1 > 0:
            dists_min -= 0.1
        dists_u = np.linspace(dists_min, dists_max + 0.1, 1000)

        recalls = []
        precisions = []
        print('getting pr...')
        for th in tqdm(dists_u, ncols=40):
            TPCount = 0
            FPCount = 0
            FNCount = 0
            TNCount = 0
            for index_q in range(dists.shape[0]):
                # Positive
                if dists[index_q, 0] < th:
                    # True
                    if np.any(np.in1d(preds[index_q, 0], gt[index_q])):
                        TPCount += 1
                    else:
                        FPCount += 1
                else:
                    if np.any(np.in1d(preds[index_q, 0], gt[index_q])):
                        FNCount += 1
                    else:
                        TNCount += 1
            assert TPCount + FPCount + FNCount + TNCount == dists.shape[0], 'Count Error!'
            if TPCount + FNCount == 0 or TPCount + FPCount == 0:
                continue
            recall = TPCount / (TPCount + FNCount)
            precision = TPCount / (TPCount + FPCount)
            recalls.append(recall)
            precisions.append(precision)

        f1_score = self.get_f1score(recalls, precisions)
        print("F1 Score: ", f1_score)

        return recall_rate

    def get_f1score(self, recalls, precisions):
        recalls = np.array(recalls)
        precisions = np.array(precisions)
        ind = np.argsort(recalls)
        recalls = recalls[ind]
        precisions = precisions[ind]
        f1s = []
        for index_j in range(len(recalls)):
            f1 = 2 * precisions[index_j] * recalls[index_j] / (precisions[index_j] + recalls[index_j])
            f1s.append(f1)

        print('f1 score: ', max(f1s))
        return f1s


if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    param_class = ModelParamsFull()
    trainer = Tester(param_class)
    trainer.do_test()

import random
from torch.utils.data import DataLoader, default_collate
from config.params_full import ModelParamsFull
from dataset.NuScenes.nuscenes_dataset import QueryDataset


def make_dataloader(working_mode, params: ModelParamsFull):
    if working_mode == 'train':
        raise NotImplementedError

    elif working_mode == 'test':
        test_dataset = QueryDataset(
            params.nusc_param.info_root[0:1],
            params.nusc_param.nusc_root,
            params.nusc_param.bev_dataset_root,
            params.nusc_param.database_index_list[0:1],
            params.nusc_param.test_query_index_list,
            params.nusc_param.gth_dist_threshold,
            l_enable=params.l_enable,
            c_enable=params.c_enable,
            r_enable=params.r_enable
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=params.test_batch_size,
            shuffle=False,
            num_workers=params.num_test_workers,
            pin_memory=True
        )

        return test_loader, test_dataset


def make_collate_fn():
    def collate_fn(batch):
        if None not in batch:
            return default_collate(batch)
        else:
            none_batch_idx = [ind for ind, value in enumerate(batch) if value is None]
            all_ind = list(range(len(batch)))
            alternate_batch_idx_list = [ind for ind in all_ind if ind not in none_batch_idx]
            alternate_batch_idx = random.sample(alternate_batch_idx_list, len(none_batch_idx))
            for i in range(len(alternate_batch_idx)):
                batch[none_batch_idx[i]] = batch[alternate_batch_idx[i]]
            return default_collate(batch)
    return collate_fn

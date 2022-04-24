from config import cfg
from dammethod.dataset.gazefollow import GazeFollowDataset,collate_fn
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

if __name__ == '__main__':

    cfg.merge_from_file("./config/gazefollow_cfg.yaml")

    train_dataset=GazeFollowDataset(cfg.DATASET.train_anno,"train",cfg,show=True)

    test_dataset=GazeFollowDataset(cfg.DATASET.test_anno,"test",cfg,show=False)

    train_loader = DataLoader(train_dataset,
                                   batch_size=cfg.DATASET.train_batch_size,
                                   num_workers=cfg.DATASET.load_workers,
                                   shuffle=True,
                                   collate_fn=collate_fn)

    test_loader = DataLoader(test_dataset,
                                 batch_size=cfg.DATASET.test_batch_size,
                                 num_workers=cfg.DATASET.load_workers,
                                 shuffle=False,
                                 collate_fn=collate_fn)

    pbar=tqdm(total=len(train_loader))
    for i ,data in enumerate(train_loader,0):

        for key,value in data.items():

            if torch.isnan(value).any():
                print(key,value)

        pbar.update(1)

    # for i in range(len(test_dataset)):
    #
    #     all_data=test_dataset.__getitem__(i)
    #
    #     for key,value in all_data.items():
    #
    #         print(key)
    #         try:
    #             print(value.shape,value.dtype)
    #         except:
    #             print(value,type(value))
    #
    #     print("------------------")


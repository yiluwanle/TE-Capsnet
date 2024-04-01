import os
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch


class MyDataAndDate(Dataset):
    def __init__(self, x, y, date):
        self.dataset_x = torch.FloatTensor(x)
        self.dateset_y = torch.FloatTensor(y)
        self.dateset_date = torch.FloatTensor(date)

    def __len__(self):
        return len(self.dataset_x)

    def __getitem__(self, item):
        return self.dataset_x[item], self.dateset_y[item], self.dateset_date[item]


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return [mae, mape, rmse]


def load_dataset_train(dataset_dir):
    train = np.load(dataset_dir + "train.npz")
    val = np.load(dataset_dir + "val.npz")
    date_train = np.load(dataset_dir + "train_date.npz")['x']
    date_val = np.load(dataset_dir + "val_date.npz")['x']
    datasets = {
        'train': MyDataAndDate(train['x'], train['y'], date_train),
        'val': MyDataAndDate(val['x'], val['y'], date_val)
    }
    return datasets


def load_dataset_test(dataset_dir):
    test = np.load(dataset_dir + "test.npz")
    date_test = np.load(dataset_dir + "test_date.npz")['x']
    datasets = {
        'test': MyDataAndDate(test['x'], test['y'], date_test)
    }
    return datasets



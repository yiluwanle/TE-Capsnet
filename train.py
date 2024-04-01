import json
import time
import torch
from torch import optim
from tqdm import tqdm
import os
import util
import numpy as np
from torch.utils.data import DataLoader
from model import Network
import yaml

with open('config.yaml', 'r', encoding='utf-8') as f:
    args = yaml.load(f.read(), Loader=yaml.FullLoader)

os.environ["CUDA_VISIBLE_DEVICES"] = args["device_list"]


def main(expid):
    device = torch.device(args["device"] if torch.cuda.is_available() else "cpu")
    print(device)

    start_time = time.time()
    data = util.load_dataset_train(args['data'])
    train_loader = DataLoader(data['train'], batch_size=args["batch_size"], shuffle=True,
                              num_workers=args["num_workers"])
    eval_loader = DataLoader(data['val'], batch_size=args["batch_size"], shuffle=True, num_workers=args["num_workers"])
    end_time = time.time()
    print(f"loader data finish, time_spent:{end_time - start_time}")

    model = Network(args=args, device=device)
    model.cuda()
    model = torch.nn.DataParallel(model)

    # 统计模型的总参数数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params}")
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer = optim.Adam(model.parameters(), lr=args["learn_rate"], weight_decay=args["weight_decay"], eps=1e-5)
    loss = util.masked_mae

    val_time = []
    train_time = []
    val_loss = []
    val_mape = []
    val_rmse = []
    tra_loss = []
    tra_mape = []
    tra_rmse = []

    for epoch in range(args["epochs"]):
        train_loss = []
        train_mape = []
        train_rmse = []
        start_time = time.time()
        iter = 1
        loop = tqdm((train_loader), desc="Training", total=len(train_loader))
        for x, y, date in loop:
            model.train()
            x, y, date = x.cuda(), y.cuda(), date.cuda()
            optimizer.zero_grad()
            predict = model(x, date)
            loss_now = loss(predict, y)
            loss_now.backward()
            optimizer.step()
            matrics = util.metric(predict, y)
            train_loss.append(loss_now.item())
            train_mape.append(matrics[1])
            train_rmse.append(matrics[2])
            if iter % args['print_every'] == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, np.mean(train_loss[iter - args["print_every"]:iter - 1]),
                                 np.mean(train_mape[iter - args["print_every"]:iter - 1]),
                                 np.mean(train_rmse[iter - args["print_every"]:iter - 1])), flush=True)
            iter += 1
            loop.set_postfix(loss=loss_now.item())
        duration = time.time() - start_time
        train_time.append(duration)

        # validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []
        s1 = time.time()
        eval_loop = tqdm(eval_loader, desc="Evaluating", total=len(eval_loader))
        for x, y, date in eval_loop:
            model.eval()
            x, y, date = x.to(device), y.to(device), date.to(device)
            predict = model(x, date)
            loss_now = loss(predict, y)
            matrics = util.metric(predict, y)
            valid_loss.append(loss_now.item())
            valid_mape.append(matrics[1])
            valid_rmse.append(matrics[2])
            eval_loop.set_postfix(loss=loss_now.item())
        s2 = time.time()
        log = 'Epoch: {:03d}, validation Time: {:.4f} secs'
        print(log.format(epoch, (s2 - s1)))
        val_time.append(s2 - s1)

        mtrain_loss = np.mean(train_loss)
        tra_loss.append(mtrain_loss)
        mtrain_mape = np.mean(train_mape)
        tra_mape.append(mtrain_mape)
        mtrain_rmse = np.mean(train_rmse)
        tra_rmse.append(mtrain_rmse)

        mvalid_loss = np.mean(valid_loss)
        val_loss.append(mvalid_loss)
        mvalid_mape = np.mean(valid_mape)
        val_mape.append(mvalid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        val_rmse.append(mvalid_rmse)
        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(epoch, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse,
                         train_time[epoch]),
              flush=True)

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    bestid = np.argmin(val_loss)
    best_epoch_road = args['save'] + 'best/exp_' + str(expid) + '_best_epoch_' + str(bestid) + '_' + str(
        round(val_loss[bestid], 2)) + ".pth"
    print(f"best_epoch_road = {best_epoch_road}")
    torch.save(model.module.state_dict(), best_epoch_road)
    args["checkpoint"] = best_epoch_road


if __name__ == "__main__":
    t1 = time.time()
    main(args["main_id"])
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
    with open(f'{args["save"]}checkpoint.txt', 'w') as f:
        f.write(args["checkpoint"])

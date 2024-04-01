import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import Network
import util
import yaml
from collections import OrderedDict

with open('config.yaml', 'r', encoding='utf-8') as f:
    args = yaml.load(f.read(), Loader=yaml.FullLoader)


def test():
    device = torch.device(args['device'])
    data = util.load_dataset_test(args['data'])
    test_loader = DataLoader(data['test'], batch_size=args["batch_size"], shuffle=True, num_workers=args["num_workers"])
    model = Network(args=args, device=device)
    model.cuda()
    model = torch.nn.DataParallel(model)

    new_state_dict = OrderedDict()
    for key, value in torch.load(args['checkpoint']).items():
        name = 'module.' + key
        new_state_dict[name] = value
    model.load_state_dict(new_state_dict)

    model.eval()

    outputs = []
    reals = []
    for x, y, date in test_loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            preds = model(x, date)
        outputs.append(preds)
        reals.append(y)

    reals = torch.cat(reals, dim=0)
    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:reals.size(0), ...]

    amae = []
    amape = []
    armse = []
    for i in range(12):
        real = reals[:, i, :]
        metrics = util.metric(yhat[:, i, :], real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[2], metrics[1] * 100))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test RMSE: {:.4f}，Test MAPE: {:.4f},'
    print(log.format(np.mean(amae), np.mean(armse), np.mean(amape)))


if __name__ == "__main__":
    test()

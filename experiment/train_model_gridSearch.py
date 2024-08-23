import sys
import scipy.stats
sys.path.append("..")
from utils.dataset_utils import DataLoader_ISM
from utils.utils import random_planetoid_splits
import torch
import numpy as np
from utils import param_utils
from utils.metric_utils import *
import os
import gc

gc.collect()
torch.cuda.empty_cache()


def RunExp(args, data, Net):
    def train(model, optimizer, data, dprate):
        model.train()
        optimizer.zero_grad()
        out = model(data)[data.train_mask]
        loss = new_rank_loss(data.y[data.train_mask], out)
        # optimizer.zero_grad()  # 清空过往梯度，在有batch的时候用
        loss.backward()
        optimizer.step()
        del out

    @torch.no_grad()
    def test(model, data):
        model.eval()
        logits = model(data)
        accs, losses, preds = [], [], []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask]
            acc = scipy.stats.kendalltau(pred.reshape(pred.size()[0]).detach().cpu().numpy(),
                                         data.y[mask].detach().cpu().numpy())[0]
            loss = new_rank_loss(data.y[mask], pred)
            if np.isnan(acc):
                print("SAD")
            accs.append(acc)
            preds.append(pred.detach().cpu())
            losses.append(loss.detach().cpu())
        return accs, preds, losses

    net = Net(data, args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    permute_masks = random_planetoid_splits
    data = permute_masks(data, args)

    model, data = net.to(device), data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_corr, test_corr = 0, 0
    best_val_loss = float('inf')
    val_loss_history = []
    train_log, val_log, test_log = [], [], []
    Gamma_0, Gamma_1 = [], []

    for epoch in range(args.epochs):
        train(model, optimizer, data, args.dprate)

        [train_corr, val_corr, tmp_test_corr], preds, [
            train_loss, val_loss, tmp_test_loss] = test(model, data)

        print("epoch:{} train_Loss:{}, train_corr:{}".format(epoch, train_loss, train_corr))

        train_log.append([epoch, train_loss, train_corr])
        val_log.append([epoch, val_loss, val_corr])
        test_log.append([epoch, tmp_test_loss, tmp_test_corr])

        if val_corr > best_val_corr:  # val_loss < best_val_loss:
            best_val_corr = val_corr
            best_val_loss = val_loss
            test_corr = tmp_test_corr
            # best_model_wts = copy.deepcopy(model.state_dict())
            print("** epoch:{}, train_loss:{}, val_loss:{}, test_loss:{}".format(epoch, train_loss, val_loss,
                                                                                 tmp_test_loss))
            print("** epoch:{}, train_cor:{}, val_cor:{}, test_corr:{}".format(epoch, train_corr, val_corr,
                                                                               tmp_test_corr))

        if epoch >= 0:
            val_loss_history.append(val_loss)
            # val_acc_history.append(val_corr)
            if 0 < args.early_stopping < epoch:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss >= tmp.mean().item():
                    break

    return test_corr, best_val_corr, Gamma_0, Gamma_1, model


if __name__ == '__main__':

    args = param_utils.parse_args()
    print("args = ", args)

    Net = param_utils.get_net(args.net)
    path = os.path.join('..', 'data', args.dataset, '{}-simplex'.format(args.hubOrder),
                        args.dataset + str(args.maxOrder) + "_result.txt")

    for time in range(args.RPMAX):
        for beta1 in [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]:
            min_acc = 0
            for learning_rate in [0.001]:
                for weight_decay in [0.0]:
                    # for beta2 in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
                    args.beta1 = beta1
                    # args.beta2 = beta2
                    args.lr = learning_rate
                    args.weight_decay = weight_decay
                    data = DataLoader_ISM(args.dataset, args)
                    print("___args.beta_____:", beta1)
                    print("___args.learning_rate___:", learning_rate)

                    test_acc, best_val_acc, Gamma_0, Gamma_1, best_model = RunExp(args, data, Net)

                    print("==============correlation==============")
                    our_tau_list, dc_tau_list, ks_tau_list, nd_tau_list, hi_tau_list, MRCNN_tau_list, bc_tau_list, vc_tau_list = compare_tau(
                        data, best_model, args.dataset, args.hubOrder)

                    # sir_list_true = data.y.cpu().numpy()
                    # sir_list_pred = best_model(data).detach().cpu().numpy()
                    if our_tau_list > min_acc:
                        min_acc = our_tau_list
                        best_test_acc = test_acc
                        best_dc_tau_list = dc_tau_list
                        best_ks_tau_list = ks_tau_list
                        best_nd_tau_list = nd_tau_list
                        best_hi_tau_list = hi_tau_list
            with open(path, 'a+') as p:
                p.write(str(time) + ',')
                p.write(str(args.lr) + ',')
                p.write(str(args.weight_decay) + ',')
                p.write(str(args.beta1) + ',')
                p.write(str(args.beta2) + ',')
                p.write(str(best_test_acc) + ',')
                p.write(str(min_acc) + ',')
                p.write(str(best_dc_tau_list) + ',')
                p.write(str(best_ks_tau_list) + ',')
                p.write(str(best_nd_tau_list) + ',')
                p.write(str(best_hi_tau_list) + ',')
                p.write(str(MRCNN_tau_list) + ',')
                p.write(str(bc_tau_list) + ',')
                p.write(str(vc_tau_list) + ',')
                p.write(str('\n'))

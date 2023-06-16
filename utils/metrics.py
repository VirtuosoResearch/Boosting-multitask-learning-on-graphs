import torch
import numpy as np
import torch.nn.functional as F
import torch_geometric
# from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from torchmetrics.functional.classification import f1_score, accuracy, auroc, average_precision

''' Evaluation metrics '''
def evaluate_roc_auc(preds, target, reduction="mean"):
    if len(target.shape) == 1:
        target = target.unsqueeze(-1)

    rocauc_list = auroc(preds, target.type(torch.long), num_classes=target.shape[1], average=None)
    if reduction == "mean":
        return rocauc_list.mean()
    else:
        if len(rocauc_list.shape) == 0:
            rocauc_list = rocauc_list.unsqueeze(-1)
        return rocauc_list

def evaluate_f1(preds, target, reduction="mean"):
    if len(target.shape) == 1:
        target = target.unsqueeze(-1)

    f1_list = f1_score(preds, target.type(torch.long), num_classes=target.shape[1], average=None)
    if reduction == "mean":
        return f1_list.mean()
    else:
        return f1_list

def evaluate_accuracy(preds, target, reduction="mean"):
    if len(target.shape) == 1:
        target = target.unsqueeze(-1)
    
    acc_list = accuracy(preds, target.type(torch.long), num_classes=target.shape[1], average=None)
    if reduction == "mean":
        return acc_list.mean()
    else:
        return acc_list

def evaluate_precision(preds, target, reduction="mean"):
    if len(target.shape) == 1:
        acc_list = average_precision(preds, target.type(torch.long), num_classes=1, average=None)
    else:
        acc_list = average_precision(preds, target.type(torch.long), num_classes=target.shape[1], average=None)
    acc_list = torch.Tensor(acc_list) if type(acc_list) == list else torch.Tensor([acc_list])
    if reduction == "mean":
        return acc_list.mean()
    else:
        return acc_list

def evaluate_mae(preds, target, reduction="mean"):
    mses = F.l1_loss(preds, target, reduction='none')
    if reduction == "mean":
        return mses.mean()
    else:
        return mses

# def evaluate_roc_auc(y_true, y_pred, reduction="mean"):
#     # average roc auc scores over all labels
#     rocauc_list = []
#     if len(y_true.shape) == 1:
#         y_true = np.expand_dims(y_true, axis=-1)

#     for i in range(y_true.shape[1]):
#         #AUC is only defined when there is at least one positive data.
#         if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
#             rocauc_list.append(roc_auc_score(y_true[:,i], y_pred[:,i]))

#     if len(rocauc_list) == 0:
#         raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

#     rocauc_list = np.array(rocauc_list)
#     if reduction == "mean":
#         return rocauc_list.mean()
#     else:
#         return rocauc_list

# def evaluate_f1(y_true, y_pred, reduction="mean"):
#     # average f1 scores over all labels
#     f1_list = []
#     if len(y_true.shape) == 1:
#         y_true = np.expand_dims(y_true, axis=-1)
#     for i in range(y_true.shape[1]):
#         tmp_f1_score = f1_score(y_true[:, i], y_pred[:, i])
#         f1_list.append(tmp_f1_score)
    
#     f1_list = np.array(f1_list)
#     if reduction == "mean":
#         return f1_list.mean()
#     else:
#         return f1_list

# def evaluate_accuracy(y_true, y_pred, reduction="mean"):
#     accuracy_list = []
#     if len(y_true.shape) == 1:
#         y_true = np.expand_dims(y_true, axis=-1)
#     for i in range(y_true.shape[1]):
#         tmp_acc = accuracy_score(y_true[:, i], y_pred[:, i])
#         accuracy_list.append(tmp_acc)

#     accuracy_list = np.array(accuracy_list)
#     if reduction == "mean":
#         return accuracy_list.mean()
#     else:
#         return accuracy_list

@torch.no_grad()
def test_longtail_performance(model, data, test_idx, thres=8):
    # Compute test node degrees
    model.eval()
    if data.edge_index is not None:
        degrees = torch_geometric.utils.degree(data.edge_index[1], data.x.size(0), dtype=data.x.dtype)
        output = model(data.x, data.edge_index)
    else:
        degrees = data.adj_t.sum(0)
        output = model(data.x, data.adj_t)
    test_degrees = degrees[test_idx]

    tmp_test_idx = test_idx[test_degrees<=thres]
    if len(tmp_test_idx) == 0:
        acc_test = -1
    else:
        acc_test = evaluate_accuracy(
            output[tmp_test_idx].detach().cpu().numpy(), 
            data.y[tmp_test_idx].detach().cpu().numpy()
            )
    return acc_test

@torch.no_grad()
def test_performance_in_degrees(model, data, test_idx, degree_power=7):
    # Compute test node degrees
    model.eval()
    if data.edge_index is not None:
        degrees = torch_geometric.utils.degree(data.edge_index[1], data.x.size(0), dtype=data.x.dtype)
        output = model(data.x, data.edge_index)
    else:
        degrees = data.adj_t.sum(0)
        output = model(data.x, data.adj_t)
    test_degrees = degrees[test_idx]

    accs = []
    degree_range = np.arange(degree_power)
    for degree in np.exp2(degree_range):
        tmp_test_idx = test_idx[torch.logical_and(int(degree/2)<test_degrees, test_degrees<=degree)]
        if len(tmp_test_idx) == 0:
            accs.append(-1)
        else:
            acc_test = evaluate_accuracy(
                output[tmp_test_idx].detach().cpu().numpy(), 
                data.y[tmp_test_idx].detach().cpu().numpy())
            accs.append(acc_test)
    return np.array(accs)
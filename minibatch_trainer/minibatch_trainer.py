import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric

from utils.metrics import evaluate_accuracy, evaluate_f1, evaluate_roc_auc, evaluate_precision, evaluate_mae
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.utils import to_scipy_sparse_matrix
from utils.pagerank import pagerank_scipy

name_to_evaluators = {
    "accuracy": evaluate_accuracy,
    "f1_score": evaluate_f1,
    "roc_auc": evaluate_roc_auc,
    "precision": evaluate_precision,
    "mae": evaluate_mae
}

names_to_criterions = {
    "multiclass": F.nll_loss,
    "multilabel": F.binary_cross_entropy_with_logits,
    "regression": F.l1_loss
}

class Trainer:
    '''
    Training logic for semi-supervised node classification
    '''
    def __init__(self, model, optimizer, data, train_loader, test_loader, device,
                epochs, log_steps, checkpoint_dir, degrees, degree_thres, criterion = "multiclass", evaluator="accuracy", monitor="avg", decoupling=False):
        self.model = model
        self.optimizer = optimizer
        self.data = data
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.evaluator = evaluator
        self.criterion = criterion
        self.device = device
        self.decoupling = decoupling
        self.input_dim = self.data.x.shape[1]

        ''' Training config '''
        self.epochs = epochs
        self.log_steps = log_steps
        self.checkpoint_dir = checkpoint_dir
        self.degrees = degrees
        self.degree_thres = degree_thres

        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.monitor = monitor

    def train_epoch(self, epoch):
        self.model.train()

        total_loss = 0; steps = 0
        if self.decoupling:
            # For decoupling trainer, the train loader only loads propagated features and labels
            for batch in self.train_loader:
                xs, y, train_mask = batch
                xs, y, train_mask = xs.to(self.device), y.to(self.device), train_mask.to(self.device)
                xs = [x for x in torch.split(xs, self.input_dim, -1)]
                self.optimizer.zero_grad()
                
                outputs = self.model(xs, return_softmax=self.criterion=="multiclass")
                labels = y.squeeze(1) if self.criterion == "multiclass"  else y
                loss = names_to_criterions[self.criterion](outputs[train_mask], labels[train_mask])
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item(); steps += 1
        else:
            for data in self.train_loader:
                self.optimizer.zero_grad()
                data = data.to(self.device)
                if hasattr(data, "edge_norm"):
                    outputs = self.model(data.x, edge_index = data.edge_index, edge_weight = data.edge_norm, return_softmax=self.criterion=="multiclass")[data.train_mask]
                    labels = data.y.squeeze(1)[data.train_mask] if self.criterion == "multiclass" else data.y[data.train_mask]
                    losses = names_to_criterions[self.criterion](outputs, labels, reduce='none')
                    loss = (losses*(data.node_norm[data.train_mask])).sum()
                else:
                    if data.adj_t is not None:
                        outputs = self.model(data.x, edge_index = data.adj_t, return_softmax=self.criterion=="multiclass")[data.train_mask]
                    else:
                        outputs = self.model(data.x, edge_index = data.edge_index, return_softmax=self.criterion=="multiclass")[data.train_mask]
                    labels = data.y.squeeze(1)[data.train_mask] if self.criterion == "multiclass"  else data.y[data.train_mask]
                    loss = names_to_criterions[self.criterion](outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item(); steps += 1
        
        return total_loss / steps

    def train(self):
        best_val_acc = test_acc = test_longtail_acc = 0

        for epoch in range(1, 1 + self.epochs):
            loss = self.train_epoch(epoch)
            log = self.test()
            train_acc, valid_acc, tmp_test_acc, train_loss, valid_loss, test_loss, valid_longtail_acc, tmp_test_longtail_acc = \
                log[f"train_{self.evaluator}"], log[f"valid_{self.evaluator}"], log[f"test_{self.evaluator}"], \
                log["train_loss"], log["valid_loss"], log["test_loss"], log[f"valid_longtail_{self.evaluator}"], log[f"test_longtail_{self.evaluator}"]

            monitor_metric = valid_acc if self.monitor == 'avg' else valid_longtail_acc
            if monitor_metric > best_val_acc:
                best_val_acc = monitor_metric
                test_acc = tmp_test_acc
                test_longtail_acc = tmp_test_longtail_acc

                ''' Save checkpoint '''
                self.save_checkpoint()

            if epoch % self.log_steps == 0:
                print(f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, train loss: {train_loss:.4f}, '
                      f'Valid: {100 * valid_acc:.2f}%, valid loss: {valid_loss:.4f}, '
                      f'Test: {100 * tmp_test_acc:.2f}%, test_loss: {test_loss:.4f}, ' 
                      f'Valid longtail: {100*valid_longtail_acc:.2f}% '
                      f'Test longtail: {100*tmp_test_longtail_acc:.2f}%')
        return test_acc, test_longtail_acc

    def save_checkpoint(self, name = "model_best"):
        model_path = os.path.join(self.checkpoint_dir, f'{name}.pth')
        torch.save(self.model.state_dict(), model_path)
        print(f"Saving current model: {name}.pth ...")

    def load_checkpoint(self):
        model_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
        self.model.load_state_dict(torch.load(model_path))
        print("Loading the best checkpoint!")

    def test(self):
        log = {}
        self.model.eval()
        
        data = self.data
        if self.decoupling:
            out = self.model.inference(data.xs, device = self.device, return_softmax = self.criterion=="multiclass")
        else:
            out = self.model.inference(data.x, test_loader = self.test_loader, device = self.device, return_softmax = self.criterion=="multiclass")

        if self.criterion == "multiclass": 
            y_pred = out.argmax(dim=-1, keepdim=True)
            y_true = data.y.squeeze(1)
        elif self.criterion == "multilabel":
            y_pred = out if self.evaluator == "roc_auc" else (out > 0).type(torch.float)
            y_true = data.y
        
        train_acc = name_to_evaluators[self.evaluator](
            target = y_true[data.train_mask],
            preds = y_pred[data.train_mask],
        ).cpu().item()
        valid_acc = name_to_evaluators[self.evaluator](
            target = y_true[data.val_mask],
            preds = y_pred[data.val_mask],
        ).cpu().item()
        test_acc = name_to_evaluators[self.evaluator](
            target = y_true[data.test_mask],
            preds = y_pred[data.test_mask],
        ).cpu().item()

        train_loss = names_to_criterions[self.criterion](out[data.train_mask], y_true[data.train_mask]).cpu().item()
        valid_loss = names_to_criterions[self.criterion](out[data.val_mask], y_true[data.val_mask]).cpu().item()
        test_loss  = names_to_criterions[self.criterion](out[data.test_mask], y_true[data.test_mask]).cpu().item()

        ''' Evaluate longtail performance'''
        # valid_degrees = self.degrees[data.val_mask]
        longtail_valid_mask = torch.logical_and(data.val_mask, self.degrees<=self.degree_thres)
        valid_longtail_acc = name_to_evaluators[self.evaluator](
            target = y_true[longtail_valid_mask],
            preds = y_pred[longtail_valid_mask],
        ).cpu().item()

        # test_degrees = self.degrees[data.test_mask]
        longtail_test_mask = torch.logical_and(data.test_mask, self.degrees<=self.degree_thres)
        test_longtail_acc = name_to_evaluators[self.evaluator](
            target = y_true[longtail_test_mask],
            preds = y_pred[longtail_test_mask],
        ).cpu().item()

        log = {
            "train_loss": train_loss, 
            "valid_loss": valid_loss, 
            "test_loss": test_loss, 
            f"train_{self.evaluator}": train_acc, 
            f"valid_{self.evaluator}": valid_acc, 
            f"test_{self.evaluator}": test_acc, 
            f"valid_longtail_{self.evaluator}": valid_longtail_acc, 
            f"test_longtail_{self.evaluator}": test_longtail_acc
        }

        return log
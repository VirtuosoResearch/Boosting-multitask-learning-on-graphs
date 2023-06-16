import torch
import torch.nn as nn
import numpy as np

class GroupedModel(nn.Module):

    def __init__(self):
        super(GroupedModel, self).__init__()
        self.grouped_models = nn.ModuleList()
        self.grouped_tasks = []

    def add_model(self, model, task_idxes):
        self.grouped_models.append(model)
        self.grouped_tasks.append(task_idxes)

    @torch.no_grad()
    def inference(self, xs_all, device, return_softmax=True):
        y_preds = []
        task_idxes = np.concatenate(self.grouped_tasks)
        assert len(self.grouped_models) == len(self.grouped_tasks)
        for model in self.grouped_models:
            out = model.inference(xs_all, device = device, return_softmax = return_softmax)
            y_preds.append(out)
        y_preds = torch.concat(y_preds, dim=1)
        y_preds[:, task_idxes] = y_preds.clone()
        return y_preds
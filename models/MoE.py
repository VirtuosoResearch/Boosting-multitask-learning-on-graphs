import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models import SIGN_MLP
from models.Precomputing_base import PrecomputingBase

class MixtureOfExperts(PrecomputingBase):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, use_bn=True, num_of_experts=10):
        super(MixtureOfExperts, self).__init__(in_channels, hidden_channels, out_channels, num_layers,
                 dropout, use_bn)
        self.num_of_experts = num_of_experts
        self.num_of_tasks = out_channels

        self.base_models = nn.ModuleList()
        for i in range(num_of_experts):
            self.base_models.append(
                SIGN_MLP(in_channels, hidden_channels, hidden_channels, num_layers, dropout, use_bn)
            )
        self.task_gates = nn.ModuleList()
        for i in range(out_channels):
            self.task_gates.append(nn.Linear(in_channels*(num_layers+1), num_of_experts, bias=False))
        self.classifier = nn.Linear(hidden_channels, 1)

    def reset_parameters(self):
        for i in range(self.num_of_experts):
            self.base_models[i].reset_parameters()
        for i in range(self.num_of_tasks):
            self.task_gates[i].reset_parameters()
        self.classifier.reset_parameters()
        

    def forward(self, xs, return_softmax=True):
        experts_outputs = []
        for i in range(self.num_of_experts):
            tmp_outputs = self.base_models[i](xs, return_softmax=False)
            experts_outputs.append(tmp_outputs)
        experts_outputs = torch.stack(experts_outputs, dim=1)

        task_outputs = []
        inputs = torch.concat(xs, dim=1)
        for i in range(self.num_of_tasks):
            tmp_task_gates = self.task_gates[i](inputs)
            tmp_task_gates = F.softmax(tmp_task_gates, dim=1).unsqueeze(-1)
            tmp_task_outputs = (tmp_task_gates * experts_outputs).sum(dim=1)
            tmp_task_outputs = self.classifier(tmp_task_outputs)
            task_outputs.append(tmp_task_outputs)
        
        task_outputs = torch.concat(task_outputs, dim=1)
        # if return_softmax:
        #     task_outputs = F.log_softmax(task_outputs, dim=1)
        return task_outputs
    
    
"""Mechanisms for Fed Average."""

import torch
from collections import OrderedDict
from .patch.module import PatchedModule

class FedAvg():
    def __init__(self, local_lr, local_loss, local_epochs, 
                 local_batchsize, setup, use_updates=True):
        self.local_lr = local_lr
        if local_loss == 'CrossEntropy':
            self.local_lossfn = torch.nn.CrossEntropyLoss().to(**setup)
        self.local_epochs = local_epochs
        self.local_batchsize = local_batchsize
        self.use_updates = use_updates

    def local_update(self, model, input_data, labels):
        """Take a few local gradient descent steps."""
        patched_model = PatchedModule(model)
        if self.use_updates:
            patched_model_origin = PatchedModule(model)
        for epoch_id in range(self.local_epochs):
            for batch_id in range(input_data.shape[0] // self.local_batchsize):
                log_probs = patched_model(input_data[batch_id * self.local_batchsize:
                                         (batch_id + 1) * self.local_batchsize], 
                                          patched_model.parameters)
                labels_ = labels[batch_id * self.local_batchsize:
                                 (batch_id + 1) * self.local_batchsize]
                
                loss = self.local_lossfn(log_probs, labels_)
                grad = torch.autograd.grad(loss, patched_model.parameters.values(),
                                   retain_graph=True, create_graph=True, only_inputs=True)
                
                patched_model.parameters = OrderedDict((name, param - self.local_lr * grad_part)
                                               for ((name, param), grad_part)
                                               in zip(patched_model.parameters.items(), grad))
        
        if self.use_updates:
            patched_model.parameters = OrderedDict((name, param - param_origin)
              for ((name, param), (name_origin, param_origin))
              in zip(patched_model.parameters.items(), patched_model_origin.parameters.items()))
        
        return list(patched_model.parameters.values())
        
        
        
        

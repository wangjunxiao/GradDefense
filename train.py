import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader



def train_one_batch(model: nn.Module,
                 train_loader: DataLoader, 
                 device: torch.device,
                 loss_fn = nn.CrossEntropyLoss()):
    
    x, y = next(iter(train_loader))
    #print('x shape is', x.shape)
    #print('y shape is', y.shape)
    
    x = x.to(device)
    y = y.to(device)
    
            
    # Compute prediction and loss
    pred = model(x)
    loss = loss_fn(pred, y)
    
    # Backward propagation
    dy_dx = torch.autograd.grad(outputs=loss, 
                                inputs=model.parameters())
    
    return list((_.detach().clone() for _ in dy_dx))
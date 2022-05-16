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



def train_by_gradient_clipping(model: nn.Module,
                     train_loader: DataLoader, 
                     device: torch.device,
                     loss_fn = nn.CrossEntropyLoss()):
    
    x, y = next(iter(train_loader))
    #print('x shape is', x.shape)
    #print('y shape is', y.shape)
    
    x = x.to(device)
    y = y.to(device)
    
    
    sample_gradients_pool = []
    for index in range(len(x)):
        # Compute prediction and loss
        pred = model(torch.unsqueeze(x[index], dim = 0))
        loss = loss_fn(pred, torch.unsqueeze(y[index], dim = 0))
    
        # Backward propagation
        dy_dx = torch.autograd.grad(outputs=loss, 
                                    inputs=model.parameters())
        
        sample_gradients_pool.append(list((_.detach().clone() for _ in dy_dx)))
    
    return sample_gradients_pool
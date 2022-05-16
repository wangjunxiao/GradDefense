import torch
import math

from train import train_one_batch
from data.dataloader import CIFAR10DataLoader as DataLoader
import model.LeNet as net

from sensitivity import compute_sens
from perturb import noise
from compensate import denoise 



'''
    example: compensation w.r.t gradients.
'''


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

random_seed = 1234
torch.manual_seed(random_seed)

aggregation_base = 10
aggregation_weight = []
for i in range(aggregation_base):
    aggregation_weight.append(1.0 / aggregation_base)

Q = 6
slices_num = 10
perturb_slices_num = 5
scale = 0.0005


    
def main():
    data_loader = DataLoader()
    model = net().to(device)
    
    # Compute layer-wise gradient sensitivity
    sensitivity = compute_sens(model = model,
                               rootset_loader = data_loader.root_set_loader,
                               device = device)
    
    gradients_pool = []
    perturbed_gradients_pool = []
    
    for grad_id in range(aggregation_base):
        # Compute gradients 
        dy_dx = train_one_batch(model = model,
                                train_loader = data_loader.train_set_loader, 
                                device = device)
        
        gradients_pool.append(dy_dx)
        
        # Slicing gradients and random perturbing 
        perturbed_dy_dx = noise(dy_dx = dy_dx, 
                                sensitivity = sensitivity,
                                slices_num = slices_num,
                                perturb_slices_num = perturb_slices_num,
                                scale = scale)
        
        perturbed_grads = []
        for layer in perturbed_dy_dx:
            layer = layer.to(device)
            perturbed_grads.append(layer)
        
        perturbed_gradients_pool.append(perturbed_grads)
        
        
    layers_num = len(gradients_pool[0])
    layer_dims_pool = []
    for layer_gradient in gradients_pool[0]:    
        layer_dims = list((_ for _ in layer_gradient.shape))
        layer_dims_pool.append(layer_dims)
    
    #print(layers_num)
    #print(layer_dims_pool)
    
    
    _gradients = []
    _gradients_perturbed = []
    for layer_index in range(layers_num):
        gradients__ = torch.zeros(layer_dims_pool[layer_index]).to(device)
        for gradients_index in range(len(gradients_pool)):
            gradients__ += gradients_pool[gradients_index][layer_index] \
                * aggregation_weight[gradients_index]
        _gradients.append(gradients__)
        
        perturbed_gradients__ = torch.zeros(layer_dims_pool[layer_index]).to(device)
        for gradients_index in range(len(perturbed_gradients_pool)):
            perturbed_gradients__ += perturbed_gradients_pool[gradients_index][layer_index] \
                * aggregation_weight[gradients_index]
        _gradients_perturbed.append(perturbed_gradients__)
        
    
    _scale = 0
    for grad_id in range(aggregation_base):
        _scale += aggregation_base * perturb_slices_num / slices_num \
            * (scale ** 2) * aggregation_weight[grad_id]
        
    
    # Compensate gradients
    gradients_compensated = denoise(gradients = _gradients_perturbed,
                                    scale = math.sqrt(_scale),
                                    Q = Q)
    
    
    for layer in range(len(gradients_compensated)):
        if layer <= 2: # Show diff F-norm 
            print(gradients_compensated[layer].shape)
            layer_compensated = gradients_compensated[layer].to(device)
            print(torch.norm(layer_compensated - _gradients[layer]))
            print(torch.norm(_gradients_perturbed[layer] - _gradients[layer]))

    
    
if __name__ == '__main__':
    main()

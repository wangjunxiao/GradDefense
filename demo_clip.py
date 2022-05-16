import torch

from train import train_by_gradient_clipping
from data.dataloader import CIFAR10DataLoader as DataLoader
import model.LeNet as net

from sensitivity import compute_sens
from clip import noise



'''
    example: clipping w.r.t gradients.
'''

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

random_seed = 1234
torch.manual_seed(random_seed)

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

    

    # Compute gradients by samples
    dy_dx = train_by_gradient_clipping(model = model,
                                       train_loader = data_loader.train_set_loader, 
                                       device = device)
        
        
    # Slicing gradients and random perturbing with clipping
    perturbed_dy_dx = noise(dy_dx = dy_dx, 
                            sensitivity = sensitivity,
                            slices_num = slices_num,
                            perturb_slices_num = perturb_slices_num,
                            noise_intensity = scale)
        
    for layer in perturbed_dy_dx:
        print(layer.shape)



if __name__ == '__main__':
    main()
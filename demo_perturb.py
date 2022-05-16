import torch
import numpy as np
import matplotlib.pyplot as plt


from train import train_one_batch
from data.dataloader import CIFAR10DataLoader as DataLoader
import model.LeNet as net

from sensitivity import compute_sens
from perturb import noise

'''
    example: random perturbation w.r.t gradients.
'''


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# cifar10
labels_map = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "forg",
    7: "horse",
    8: "ship",
    9: "truck"}

random_seed = 1234
torch.manual_seed(random_seed)

is_plot = False

slices_num = 10
perturb_slices_num = 5
scale = 0.0005


def main():
    data_loader = DataLoader()
    
    # Display root_set
    print(f"Rootset size is {len(data_loader.root_set)}")
    if is_plot:
        figure = plt.figure(figsize=(12, 8))
        cols, rows = 10, 5
        for i in range(1, cols * rows+1):
            img, label = data_loader.root_set[i-1]
            figure.add_subplot(rows, cols, i)
            plt.title(labels_map[label])
            plt.axis("off")
            plt.imshow(np.transpose(img.numpy(), (1,2,0)))
        plt.show() 


    model = net().to(device)
    print('--------- model ------------')
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())
    print('----------------------------')
    
    # Compute layer-wise gradient sensitivity
    sensitivity = compute_sens(model = model,
                               rootset_loader = data_loader.root_set_loader,
                               device = device)
    
    # Compute original gradients
    dy_dx = train_one_batch(model = model,
                     train_loader = data_loader.train_set_loader, 
                     device = device)
    
    
    # Slicing gradients and random perturbing 
    perturbed_dy_dx = noise(dy_dx = dy_dx, 
                            sensitivity = sensitivity,
                            slices_num = slices_num,
                            perturb_slices_num = perturb_slices_num,
                            scale = scale)
    
    for layer in perturbed_dy_dx:
        print(layer.shape)
        
    
    
if __name__ == '__main__':
    main()


import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from data.dataloader import CIFAR10DataLoader as DataLoader
import model.LeNet as net
from utils import label_to_onehot, cross_entropy_for_onehot

from sensitivity import compute_sens
from perturb import noise


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

# cifar10 trainset 0~49999
img_ids = [7]
random_seed = 1234
torch.manual_seed(random_seed)
iteration_steps = 300

is_plot1 = False
is_plot2 = False
is_plot3 = True



def main():
    data_loader = DataLoader()
    img, label = data_loader.train_set[img_ids[0]]
    
    imgs = torch.unsqueeze(img, dim=0)
    labels = torch.Tensor([label]).long()
    for img_index in range(len(img_ids)):    
        if img_index == 0: continue
        img, label = data_loader.train_set[img_ids[img_index]]
        img = torch.unsqueeze(img, dim=0)
        # int -> tensor(long)
        label = torch.Tensor([label]).long()
        
        imgs = torch.cat((imgs, img), dim=0)
        labels = torch.cat((labels, label), dim=0)
    
    print(imgs.shape, labels.shape)

    if is_plot1:
        img_index = 0
        plt.title('img id'+str(img_ids[img_index])+'-'+ \
                  labels_map[int(labels[img_index].numpy())])
        plt.axis("off")
        plt.imshow(np.transpose(imgs[0].numpy(), (1,2,0)))
        
    x = imgs.to(device)
    # tensor(long) -> one_hot
    y = label_to_onehot(labels).to(device)
    print('imgs shape is', x.shape)
    print('labels shape is', y.shape)

    # Compute prediction and loss
    model = net().to(device)
    pred = model(x)
    print('logits shape is', pred.shape)
    loss_fn = cross_entropy_for_onehot
    loss = loss_fn(pred, y)
    
    
    
    # Compute layer-wise gradient sensitivity
    sensitivity = compute_sens(model = model,
                               rootset_loader = data_loader.root_set_loader,
                               device = device)
    
    
    
    
    # Compute original gradient 
    dy_dx = torch.autograd.grad(outputs=loss, 
                                inputs=model.parameters())
    original_dy_dx = list((_.detach().clone() for _ in dy_dx))
    
    
    
    
    
    # Slicing gradients and random perturbing 
    perturbed_gradients = noise(dy_dx = original_dy_dx, 
                                sensitivity = sensitivity,
                                slices_num = 10,
                                perturb_slices_num = 5,
                                scale = 0.1)
    
    original_dy_dx = []
    for layer in perturbed_gradients:
        layer = layer.to(device)
        original_dy_dx.append(layer)
    
    
    
    
    
    # Generate dummy data and label
    dummy_data = torch.randn(x.shape)
    dummy_label = torch.randn(y.shape)
    
    if is_plot2:
        plt.title('dummy img')
        plt.axis("off")
        plt.imshow(np.transpose(dummy_data[0].numpy(), (1,2,0)))
    
    dummy_data = dummy_data.to(device).requires_grad_(True)
    dummy_label = dummy_label.to(device).requires_grad_(True)
    
    # Reconstruction in DLG
    optimizer = torch.optim.LBFGS([dummy_data, dummy_label])
    for iters in range(iteration_steps):
        def closure():
            optimizer.zero_grad()

            dummy_pred = model(dummy_data) 
            # Regularization
            dummy_label_ = F.softmax(dummy_label, dim=-1)
            dummy_loss = loss_fn(dummy_pred, dummy_label_) 
            # Create_graph for LBFGS
            dummy_dy_dx = torch.autograd.grad(outputs=dummy_loss, 
                                              inputs=model.parameters(), 
                                              create_graph=True)
        
            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx): 
                grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()
            return grad_diff
        
        optimizer.step(closure)
        if iters % 10 == 0: 
            current_loss = closure()
            print(iters, "%.4f" % current_loss.item())
    
    if is_plot3:
        plt.title('reconstructed img')
        plt.axis("off")
        plt.imshow(np.transpose(dummy_data[0].detach().cpu().numpy(), (1,2,0)))
    
    
if __name__ == '__main__':
    main()
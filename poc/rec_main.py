"""Reconstruction experiments."""

import torch
torch.backends.cudnn.benchmark = True
import torchvision

import core
import core.options
import core.loader.modelloader as modelloader
import core.loader.dataloader as dataloader
import core.trainer.modeltrainer as modeltrainer
from core.fl.fedavg import FedAvg
from core.reconstruction.fedreconstructor import FedAvgRec
import core.defense.perturbation as perturbation

from collections import defaultdict
import datetime
import time
import os
import json
import hashlib
import random
import math

import matplotlib.pyplot as plt
from PIL import Image

# Parse input arguments
parser = core.options.parse_options()
args = parser.parse_args()


# Settings
args.model = 'ConvNet64'
args.trained_model = False
args.trained_epochs = 120
args.trained_optimizer = 'SGD'
args.trained_scheduler = 'linear'
args.trained_warmup = False
args.trained_lr = 0.1
args.trained_decay = 5e-4
args.sanity_check = False # Validate model accuracy

args.dataset = 'CIFAR10'
args.trained_batchsize = 128
args.num_classes = 10
args.num_channels = 3

args.dtype = 'float'
args.label_flip = False

args.rec_restarts = 1
args.scoring_choice = 'loss'
args.target_id = 12 # image index in validation dataset
args.num_images = 1
args.local_epochs = 1
args.local_batchsize = 1
args.local_lr = 1e-2
args.local_loss = 'CrossEntropy'

args.max_iterations = 800
args.rec_optimizer = 'adam'
args.rec_lossfn = 'sim'
args.rec_lr = 0.1
args.predict_labels = True
args.init = 'randn'
args.indices = 'def'
args.weights = 'equal'

args.tv = 1e-4
args.signed = False
args.boxed = False

args.defense = True
args.pruning_rate = 70

args.data_path = '~/.torch'
args.save_image = True
args.model_path = 'model/'

args.deterministic = True # 100% reproducibility?
if args.deterministic:
    core.utils.set_deterministic()


if __name__ == "__main__":
    torch.cuda.empty_cache()
    
    # Choose GPU device and print status information:
    setup = core.utils.system_startup()
    start_time = time.time()
    # Collect data set:
    _dm, _ds, loss_fn, trainloader, validloader = dataloader.load_dataset(dataset=args.dataset, 
                                                                          data_path=args.data_path, 
                                                                          batchsize=args.trained_batchsize)
    dm = torch.as_tensor(_dm, **setup)[:, None, None]
    ds = torch.as_tensor(_ds, **setup)[:, None, None]
    print('-------------------------------')
    # Collect model:
    model, model_seed = modelloader.load_model(model=args.model, 
                                               num_classes=args.num_classes, 
                                               num_channels=args.num_channels)
    model.to(**setup)
    model.eval()
    print('-------------------------------')
    # Load a trained model?
    if args.trained_model:
        file = f'{args.model}_{args.epochs}.pth'
        try:
            model.load_state_dict(torch.load(os.path.join(args.model_path, file), 
                                             map_location=setup['device']))
            print(f'Model loaded from file {file}.')
        except FileNotFoundError:
            print('Training the model ...')
            modeltrainer.train_model(model, loss_fn, trainloader, validloader, args, setup=setup)
            torch.save(model.state_dict(), os.path.join(args.model_path, file))
    # Sanity check: Validate model accuracy
    if args.sanity_check:
        training_stats = defaultdict(list)
        modeltrainer.validate_model(model, loss_fn, validloader, args, setup, training_stats)
        name, format = loss_fn.metric()
        print(f'Val loss is {training_stats["valid_losses"][-1]:6.4f}, \
              Val {name}: {training_stats["valid_" + name][-1]:{format}}.')
        print('-------------------------------')
    
    
    # collect reconstruction config
    rec_config = dict(init=args.init,
                      signed=args.signed,
                      boxed=args.boxed,
                      rec_lossfn=args.rec_lossfn,
                      predict_labels=args.predict_labels,
                      indices=args.indices,
                      weights=args.weights,
                      lr=args.rec_lr if args.rec_lr is not None else 0.1,
                      optimizer=args.rec_optimizer,
                      restarts=args.rec_restarts,
                      max_iterations=args.max_iterations,
                      total_variation=args.tv,
                      filter='none',
                      lr_decay=True,
                      scoring_choice=args.scoring_choice)
   
    # hash config
    config_comp = rec_config.copy()
    config_hash = hashlib.md5(json.dumps(config_comp, sort_keys=True).encode()).hexdigest()
    print(config_comp)
    print('-------------------------------')
    # collect reconstruction ground_truth and labels
    target_id = args.target_id
    ground_truth, labels = [], []
    while len(labels) < args.num_images:
        ground_truth_, label = validloader.dataset[target_id]
        if args.label_flip:
            label = random.randint(0, args.num_classes)
        if label not in labels:
            ground_truth.append(ground_truth_.to(**setup))
            labels.append(torch.as_tensor((label,), device=setup['device']))
        #ground_truth.append(ground_truth_.to(**setup))
        #labels.append(torch.as_tensor((label,), device=setup['device']))
        target_id += 1
    ground_truth = torch.stack(ground_truth)
    labels = torch.cat(labels)
    
    ground_truth.requires_grad = True

    # Run reconstruction
    fedavg = FedAvg(local_lr=args.local_lr, local_loss=args.local_loss, 
                                   local_epochs=args.local_epochs, 
                                   local_batchsize=args.local_batchsize,
                                   setup=setup,
                                   use_updates=True)
    
    parameters = fedavg.local_update(model=model, input_data=ground_truth, labels=labels) 
    parameters = [p.detach() for p in parameters]
    
    if args.defense:
        #Run defense
        parameters = perturbation.fc_perturb(parameters=parameters,  
                                             model=model,
                                             ground_truth=ground_truth,
                                             pruning_rate=args.pruning_rate,
                                             setup=setup)
                                                  
        
    # Run reconstruction in different precision?
    if args.dtype != 'float':
        if args.dtype in ['double', 'float64']:
            setup['dtype'] = torch.double
        elif args.dtype in ['half', 'float16']:
            setup['dtype'] = torch.half
        else:
            raise ValueError(f'Unknown data type argument {args.dtype}.')
        print(f'Model and input parameter moved to {args.dtype}-precision.')
        ground_truth = ground_truth.to(**setup)
        dm = torch.as_tensor(_dm, **setup)[:, None, None]
        ds = torch.as_tensor(_ds, **setup)[:, None, None]
        parameters = [g.to(**setup) for g in parameters]
        model.to(**setup)
        model.eval()
    
    mean_std = (ds, dm)
    reconstructor = FedAvgRec(model=model, labels=labels, 
                                        num_images=args.num_images, mean_std=mean_std, 
                                        rec_config=rec_config, fedavg=fedavg)

    img_shape = (args.num_channels, ground_truth.shape[2], ground_truth.shape[3])
    output, stats = reconstructor.reconstruct(parameters=parameters, 
                                              labels=labels, img_shape=img_shape)
        
    
    # Save the resulting image
    os.makedirs('results', exist_ok=True)
    os.makedirs(f'results/{config_hash}', exist_ok=True)
    
    if args.save_image:
        output_denormalized = torch.clamp(output * ds + dm, 0, 1)
        gt_denormalized = torch.clamp(ground_truth * ds + dm, 0, 1)
        for i in range(args.num_images):
            rec_filename = (f'{i}_rec.png')
            torchvision.utils.save_image(output_denormalized[i:i + 1, ...],
                                         os.path.join(f'results/{config_hash}', rec_filename))
            
            gt_filename = (f'{i}_ground_truth.png')
            torchvision.utils.save_image(gt_denormalized[i:i + 1, ...],
                                         os.path.join(f'results/{config_hash}', gt_filename))
        
        # plot the resulting image into one pdf
        gt_name = 'ground_truth'
        rec_name = 'rec'
        col_num = 10
        row_num = math.ceil(args.num_images/col_num)
        for i in range(args.num_images):
            gt_file = Image.open(os.path.join(f'results/{config_hash}/', f'{i}_{gt_name}.png'))
            rec_file = Image.open(os.path.join(f'results/{config_hash}/', f'{i}_{rec_name}.png'))
            plt.subplot(2*row_num,col_num,i+1), plt.title(f'{labels[i]}')
            plt.imshow(gt_file), plt.axis('off')
            plt.subplot(2*row_num,col_num,row_num*col_num+i+1)
            plt.imshow(rec_file), plt.axis('off')
            plt.savefig(f'results/{config_hash}/plot_result.pdf')

    # Print final timestamp
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print('---------------------------------------------------')
    print(f'Finished computations with time: \
                  {str(datetime.timedelta(seconds=time.time() - start_time))}')
    print('-------------Job finished.-------------------------')
    
"""Various utilities."""
import torch
import random
import numpy as np

import socket
import datetime


def system_startup(args=None):
    """Print useful system information."""
    print('-------------------------------') # Current time
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print('-------------------------------')
    # Choose GPU device and print status information:
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    setup = dict(device=device, dtype=torch.float)
    print(f'CPUs: {torch.get_num_threads()}, GPUs: {torch.cuda.device_count()} on {socket.gethostname()}.')
    if torch.cuda.is_available():
        print(f'GPU : {torch.cuda.get_device_name(device=device)}')
    print('-------------------------------')
    if args is not None:
        print(args)
        print('-------------------------------')
    return setup


def set_deterministic():
    """Switch pytorch into a deterministic computation mode."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def set_random_seed(seed=0):
    """sets the seed for generating random numbers."""
    torch.manual_seed(seed + 1) 
    """sets the seed for generating random numbers for the current GPU. 
        It’s safe to call this function if CUDA is not available; 
        in that case, it is silently ignored."""
    torch.cuda.manual_seed(seed + 2)
    """sets the seed for generating random numbers on all GPUs. 
        It’s safe to call this function if CUDA is not available; 
        in that case, it is silently ignored."""
    torch.cuda.manual_seed_all(seed + 3)
    
    random.seed(seed + 4)
    np.random.seed(seed + 5)
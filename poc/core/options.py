"""Parser options."""

import argparse

def parse_options():
    """Construct the central argument parser, filled with useful defaults."""
    parser = argparse.ArgumentParser(description='Reconstruct some image from a trained model.')

    # Central:
    parser.add_argument('--model', default='ConvNet64', type=str, help='Vision model.')
    parser.add_argument('--trained_model', action='store_true', help='Use a trained model.')
    parser.add_argument('--trained_epochs', default=120, type=int, help='If using a trained model, how many epochs was it trained?')
    parser.add_argument('--trained_optimizer', default='SGD', type=str, help='If using a trained model, which optimizer used')
    parser.add_argument('--trained_scheduler', default='linear', type=str, help='If using a trained model, which scheduler used?')
    parser.add_argument('--trained_warmup', action='store_true', help='If using a trained model, whether warmup used')
    parser.add_argument('--trained_lr', default=0.1, type=float, help='If using a trained model, learning rate used')
    parser.add_argument('--trained_decay', default=5e-4, type=float, help='If using a trained model, weight decay of learning rate used')
    parser.add_argument('--sanity_check', action='store_true', help='If using a trained model, whether test trained model')
    
    parser.add_argument('--dataset', default='CIFAR10', type=str, help='Vision dataset.')
    parser.add_argument('--trained_batchsize', default=128, type=int, help='If using a trained model, how many batchsize used')
    parser.add_argument('--num_classes', default=10, type=int, help='Number of classes in vision dataset.')
    parser.add_argument('--num_channels', default=3, type=int, help='Number of channels in vision dataset.')
    
    # Rec. parameters
    parser.add_argument('--dtype', default='float', type=str, help='Data type used during reconstruction [Not during training!].')
    parser.add_argument('--label_flip', action='store_true', help='Dishonest server permuting weights in classification layer.')
    
    parser.add_argument('--rec_restarts', default=1, type=int, help='How many restart resconstructions.')
    parser.add_argument('--scoring_choice', default='loss', type=str, help='How to find the best image between all restarts.')
    parser.add_argument('--target_id', default=0, type=int, help='Validation image used for reconstruction.')
    
    parser.add_argument('--num_images', default=1, type=int, help='Number of images should be recovered from the given gradients/weights.')
    parser.add_argument('--local_epochs', default=1, type=int, help='Number of epoches for federated averaging.')
    parser.add_argument('--local_batchsize', default=1, type=int, help='Number of mini batch size for federated averaging.')
    parser.add_argument('--local_lr', default=1e-4, type=str, help='Choice of learning rate used for federated averaging.')
    parser.add_argument('--local_loss', default='CrossEntropy', type=str, help='Choice of loss function for federated averaging.')
    
    parser.add_argument('--max_iterations', default=4800, type=int, help='Maximum number of iterations for reconstruction.')
    parser.add_argument('--rec_optimizer', default='adam', type=str, help='Choice of optimizer used for reconstruction.')
    parser.add_argument('--rec_lossfn', default='sim', type=str, help='Choice of loss fn for reconstruction.')
    parser.add_argument('--rec_lr', default=0.1, type=str, help='Choice of learning rate used for reconstruction.')
    parser.add_argument('--predict_labels', action='store_true', help='Predict labels.')
    parser.add_argument('--init', default='randn', type=str, help='Choice of image initialization.')
    parser.add_argument('--indices', default='def', type=str, help='Choice of indices from the parameter list.')
    parser.add_argument('--weights', default='equal', type=str, help='Choice of weighing the parameter list.')
    
    parser.add_argument('--tv', default=1e-4, type=float, help='Weight of TV penalty.')
    parser.add_argument('--signed', action='store_true', help='Use signed gradients.')
    parser.add_argument('--boxed', action='store_true', help='Use box constraints.')
    
    # Defense parameters
    parser.add_argument('--defense', action='store_true', help='Use a defense.')
    parser.add_argument('--pruning_rate', default=60, type=float, help='pruning rate for defense.')
    
    # Files and folders:
    parser.add_argument('--data_path', default='~/.torch', type=str)
    parser.add_argument('--save_image', action='store_true', help='Save the output to a file.')
    parser.add_argument('--model_path', default='models/', type=str)

    # Debugging:
    parser.add_argument('--deterministic', action='store_true', help='Disable CUDNN non-determinism.')
 
    return parser

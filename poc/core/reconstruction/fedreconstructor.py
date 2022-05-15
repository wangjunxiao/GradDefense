"""Mechanisms for image reconstruction from parameter gradients."""

import torch
from collections import defaultdict
from .metrics import total_variation as TV
from .metrics import InceptionScore
from .medianfilt import MedianPool2d

import time

class FedAvgRec():
    """Reconstruct an image after n gradient descent steps."""

    def __init__(self, model, labels, num_images, mean_std, rec_config, fedavg):
        """Initialize."""
        self.model = model
        self.setup = dict(device=next(model.parameters()).device, 
                          dtype=next(model.parameters()).dtype)
        self.rec_config = rec_config
        if self.rec_config['scoring_choice'] == 'inception':
            self.inception = InceptionScore(batch_size=1, setup=self.setup)
        self.num_images = num_images
        self.mean_std = mean_std     
        self.fedavg = fedavg
        
    def reconstruct(self, parameters, labels, img_shape, eval=True):
        """Reconstruct image from input parameters."""
        start_time = time.time()
        if eval:
            self.model.eval()
            
        # init x and scores
        x = self._init_images(img_shape)
        scores = torch.zeros(self.rec_config['restarts'])
        
        # labels
        if labels is None:
            self.reconstruct_label = True
            if self.rec_config['predict_labels'] == True:
                print('do predict labels here ... ...')
                self.reconstruct_label = False
        else:
            assert labels.shape[0] == self.num_images
            self.reconstruct_label = False
        
        # run trial and score 
        try:
            for trial in range(self.rec_config['restarts']):
                x_trial, labels = self._run_trial(x[trial], parameters, labels)
                scores[trial] = self._score_trial(x_trial, parameters, labels)
                x[trial] = x_trial
        except KeyboardInterrupt:
            print('Trial procedure manually interruped.')
            pass

        # Choose optimal result:
        stats = defaultdict(list)
        if self.rec_config['scoring_choice'] in ['pixelmean', 'pixelmedian']:
            x_optimal, stats = self._average_trials(x, labels, parameters, stats)
        else:
            print('Choosing optimal result ...')
            scores = scores[torch.isfinite(scores)]  # guard against NaN/-Inf scores?
            optimal_index = torch.argmin(scores)
            print(f'Optimal result score: {scores[optimal_index]:2.4f}')
            stats['opt'] = scores[optimal_index].item()
            x_optimal = x[optimal_index]

        print(f'Total time: {time.time()-start_time}.')
        return x_optimal.detach(), stats
    
    def _init_images(self, img_shape):
        if self.rec_config['init'] == 'randn':
            return torch.randn((self.rec_config['restarts'], self.num_images, *img_shape), 
                               **self.setup)
        elif self.rec_config['init'] == 'rand':
            return (torch.rand((self.rec_config['restarts'], self.num_images, *img_shape), 
                               **self.setup) - 0.5) * 2
        elif self.rec_config['init'] == 'zeros':
            return torch.zeros((self.rec_config['restarts'], self.num_images, *img_shape), 
                               **self.setup)
        else:
            raise ValueError()
 
    def _run_trial(self, x_trial, parameters, labels, dryrun=False):
        x_trial.requires_grad = True
        if self.reconstruct_label:
            output_test = self.model(x_trial)
            labels = torch.randn(output_test.shape[1]).to(**self.setup).requires_grad_(True)

            if self.rec_config['optimizer'] == 'adam':
                optimizer = torch.optim.Adam([x_trial, labels], lr=self.rec_config['lr'])
            elif self.rec_config['optimizer'] == 'sgd':  # actually gd
                optimizer = torch.optim.SGD([x_trial, labels], lr=0.01, 
                                            momentum=0.9, nesterov=True)
            elif self.rec_config['optimizer'] == 'LBFGS':
                optimizer = torch.optim.LBFGS([x_trial, labels])
            else:
                raise ValueError()
        else:
            if self.rec_config['optimizer'] == 'adam':
                optimizer = torch.optim.Adam([x_trial], lr=self.rec_config['lr'])
            elif self.rec_config['optimizer'] == 'SGD':  # actually gd
                optimizer = torch.optim.SGD([x_trial], lr=0.01, momentum=0.9, nesterov=True)
            elif self.rec_config['optimizer'] == 'LBFGS':
                optimizer = torch.optim.LBFGS([x_trial])
            else:
                raise ValueError()
        
        max_iterations = self.rec_config['max_iterations']
        if self.rec_config['lr_decay']:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                        milestones=[max_iterations // 2.667, max_iterations // 1.6,
                                    max_iterations // 1.142], gamma=0.1)   # 3/8 5/8 7/8
        
        try:
            for iteration in range(max_iterations):
                closure = self._gradient_closure(optimizer, x_trial, parameters, labels)
                rec_loss = optimizer.step(closure)
                if self.rec_config['lr_decay']:
                    scheduler.step()

                with torch.no_grad():
                    # Project into image space
                    if self.rec_config['boxed']:
                        (dm, ds) = self.mean_std
                        x_trial.data = torch.max(torch.min(x_trial, (1 - dm) / ds), -dm / ds)

                    if (iteration + 1 == max_iterations) or iteration % 100 == 0:
                        print(f'It: {iteration}. Rec. loss: {rec_loss.item():2.4f}.')

                    if (iteration + 1) % 500 == 0:
                        if self.rec_config['filter'] == 'none':
                            pass
                        elif self.rec_config['filter'] == 'median':
                            x_trial.data = MedianPool2d(kernel_size=3, stride=1, padding=1, 
                                                        same=False)(x_trial)
                        else:
                            raise ValueError()
        except KeyboardInterrupt:
            print(f'Recovery interrupted manually in iteration {iteration}!')
            pass
        return x_trial.detach(), labels    
 
    
    def _gradient_closure(self, optimizer, x_trial, parameters, labels):
        def closure():
            optimizer.zero_grad()
            self.model.zero_grad()
            parameters_trial = self.fedavg.local_update(self.model, x_trial, labels)
            rec_loss = reconstruction_loss(parameters_trial, parameters,
                                            rec_lossfn=self.rec_config['rec_lossfn'], 
                                            indices=self.rec_config['indices'],
                                            weights=self.rec_config['weights'])

            if self.rec_config['total_variation'] > 0:
                rec_loss += self.rec_config['total_variation'] * TV(x_trial)
            rec_loss.backward()
            if self.rec_config['signed']:
                x_trial.grad.sign_()
            return rec_loss
        return closure


    def _score_trial(self, x_trial, parameters, labels):
        if self.rec_config['scoring_choice'] == 'loss':
            self.model.zero_grad()
            parameters_trial = self.fedavg.local_update(self.model, x_trial, labels)
            return reconstruction_loss(parameters_trial, parameters,
                                        rec_lossfn=self.rec_config['rec_lossfn'], 
                                        indices=self.rec_config['indices'],
                                        weights=self.rec_config['weights'])
        elif self.rec_config['scoring_choice'] == 'tv':
            return TV(x_trial)
        elif self.rec_config['scoring_choice'] == 'inception':
            # We do not care about diversity here!
            return self.inception(x_trial)
        
        
    def _average_trials(self, x, labels, parameters, stats):
        print(f'Computing a combined result via {self.config["scoring_choice"]} ...')
        if self.config['scoring_choice'] == 'pixelmedian':
            x_optimal, _ = x.median(dim=0, keepdims=False)
        elif self.config['scoring_choice'] == 'pixelmean':
            x_optimal = x.mean(dim=0, keepdims=False)
        if self.reconstruct_label:
            labels = self.model(x_optimal).softmax(dim=1)
        self.model.zero_grad()
        parameters_trial = self.fedavg.local_update(self.model, x_optimal, labels)
        stats['opt'] = reconstruction_loss(parameters_trial, parameters,
                                        rec_lossfn=self.rec_config['rec_lossfn'], 
                                        indices=self.rec_config['indices'],
                                        weights=self.rec_config['weights'])
        print(f'Optimal result score: {stats["opt"]:2.4f}')
        return x_optimal, stats


def reconstruction_loss(parameters_trial, parameters, rec_lossfn='l2', 
                         indices='def', weights='equal'):
    """Input gradient is given data."""
    if isinstance(indices, list):
        pass
    elif indices == 'def':
        indices = torch.arange(len(parameters))
    elif indices == 'batch':
        indices = torch.randperm(len(parameters))[:8]
    elif indices == 'topk-1':
        _, indices = torch.topk(torch.stack([p.norm() for p in parameters], dim=0), 4)
    elif indices == 'top10':
        _, indices = torch.topk(torch.stack([p.norm() for p in parameters], dim=0), 10)
    elif indices == 'top50':
        _, indices = torch.topk(torch.stack([p.norm() for p in parameters], dim=0), 50)
    elif indices in ['first', 'first4']:
        indices = torch.arange(0, 4)
    elif indices == 'first5':
        indices = torch.arange(0, 5)
    elif indices == 'first10':
        indices = torch.arange(0, 10)
    elif indices == 'first50':
        indices = torch.arange(0, 50)
    elif indices == 'last5':
        indices = torch.arange(len(parameters))[-5:]
    elif indices == 'last10':
        indices = torch.arange(len(parameters))[-10:]
    elif indices == 'last50':
        indices = torch.arange(len(parameters))[-50:]
    elif indices == 'fc':
        indices = torch.arange(len(parameters))[-2:]
    elif indices == 'rl':
        indices = torch.arange(len(parameters))[:-2]
    elif indices == 'first-conv':
        indices = torch.arange(len(parameters))[:2]
    elif indices == 'all-conv':
        indices = torch.cat((torch.arange(len(parameters))[:-2:4], 
                            torch.arange(len(parameters))[1:-2:4]), 0)
    elif indices == 'last2-conv':
        indices = torch.cat((torch.arange(len(parameters))[24:-2:4], 
                            torch.arange(len(parameters))[25:-2:4]), 0)
    else:
        raise ValueError()
    
    setup = parameters[0]
    if weights == 'linear':
        weights = torch.arange(len(parameters), 0, -1, dtype=setup.dtype, device=setup.device) / len(parameters)
    elif weights == 'exp':
        weights = torch.arange(len(parameters), 0, -1, dtype=setup.dtype, device=setup.device)
        weights = weights.softmax(dim=0)
        weights = weights / weights[0]
    else:
        weights = setup.new_ones(len(parameters))

    pnorm = [0, 0]
    loss = 0
    total_loss = 0
    if indices == 'topk-2':
        _, indices = torch.topk(torch.stack([p.norm().detach() for p in parameters_trial], dim=0), 4)
    for i in indices:
        if rec_lossfn == 'l2':
            loss += ((parameters_trial[i] - parameters[i]).pow(2)).sum() * weights[i]
        elif rec_lossfn == 'l1':
            loss += ((parameters_trial[i] - parameters[i]).abs()).sum() * weights[i]
        elif rec_lossfn == 'max':
            loss += ((parameters_trial[i] - parameters[i]).abs()).max() * weights[i]
        elif rec_lossfn == 'sim':
            loss -= (parameters_trial[i] * parameters[i]).sum() * weights[i]
            pnorm[0] += parameters_trial[i].pow(2).sum() * weights[i]
            pnorm[1] += parameters[i].pow(2).sum() * weights[i]
        elif rec_lossfn == 'simlocal':
            loss += 1 - torch.nn.functional.cosine_similarity(parameters_trial[i].flatten(),
                                                              parameters[i].flatten(),
                                                              0, 1e-10) * weights[i]
    if rec_lossfn == 'sim':
        loss = 1 + loss / pnorm[0].sqrt() / pnorm[1].sqrt()

        # Accumulate final costs
        total_loss += loss
    return total_loss

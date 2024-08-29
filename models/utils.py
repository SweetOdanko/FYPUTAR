import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def initialize_weights(model):

    if isinstance(model, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(model.weight)
        if model.bias is not None:
            model.bias.data.fill_(0.0)
    elif isinstance(model, torch.nn.Linear):
        torch.nn.init.xavier_normal_(model.weight)
        model.bias.data.fill_(0.0)

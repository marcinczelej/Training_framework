class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, batch_accumulation=1):
        self.batch_accumulation = batch_accumulation
        self.reset()

    def reset(self):
        self.current = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.current = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (self.count * self.batch_accumulation)

from torch.optim.lr_scheduler import _LRScheduler

class ExponentialCyclicLR(_LRScheduler):
    """
    Exponential cyclic learning rate scheduler.
    - Increases exponentially from initial learning rate to max_lr over step_size_up iterations.
    - Decreases exponentially from max_lr to initial learning rate over step_size_down iterations.
    - Total cycle length = step_size_up + step_size_down.
    - Repeats this cycle indefinitely.
    """
    def __init__(self, optimizer, max_lr, step_size_up, step_size_down, last_epoch=-1):
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down
        super().__init__(optimizer, last_epoch)
        # Each param group should have 'initial_lr' set (PyTorch does this automatically)

    def get_lr(self):
        lrs = []
        cycle_pos = self.last_epoch % (self.step_size_up + self.step_size_down)
        for base_lr in self.base_lrs:
            if cycle_pos < self.step_size_up:
                # Exponential increase
                factor = (self.max_lr / base_lr) ** (cycle_pos / self.step_size_up)
                lr = base_lr * factor
            else:
                # Exponential decrease
                factor = (base_lr / self.max_lr) ** ((cycle_pos - self.step_size_up) / self.step_size_down)
                lr = self.max_lr * factor
            lrs.append(lr)
        return lrs

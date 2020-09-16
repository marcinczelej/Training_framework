import math
from torch.optim.lr_scheduler import LambdaLR

"""
    flat + cosine schedule
"""


def get_flat_cosine_schedule(optimizer, num_training_steps, percentege_of_const=0.7, num_cycles=0.5, last_epoch=-1):
    """
    Before percentage_of_const get constant lr,
    then do cosine decay till ends
    """
    # print("num_training_steps = ", num_training_steps)
    # print("int(num_training_steps*percentege_of_const) = ", int(num_training_steps*percentege_of_const))

    def lr_lambda(current_step):
        if current_step < int(num_training_steps * percentege_of_const):
            # print(current_step, " returning 1")
            return 1.0
        progress = float(current_step - num_training_steps * percentege_of_const) / float(
            max(1, num_training_steps - num_training_steps * percentege_of_const)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

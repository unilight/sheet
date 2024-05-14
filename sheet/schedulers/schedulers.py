import copy

from torch.optim.lr_scheduler import MultiStepLR, StepLR

# Reference: https://github.com/s3prl/s3prl/blob/master/s3prl/schedulers.py


def get_scheduler(optimizer, scheduler_name, total_steps, scheduler_config):
    scheduler_config = copy.deepcopy(scheduler_config)
    scheduler = eval(f"get_{scheduler_name}")(
        optimizer, num_training_steps=total_steps, **scheduler_config
    )
    return scheduler


def get_multistep(optimizer, num_training_steps, milestones, gamma):
    return MultiStepLR(optimizer, milestones, gamma)


def get_stepLR(optimizer, num_training_steps, step_size, gamma):
    return StepLR(optimizer, step_size, gamma)

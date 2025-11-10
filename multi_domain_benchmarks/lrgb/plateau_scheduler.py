
import math

import torch.optim as optim
from torch.optim import Optimizer


def scheduler_reduce_on_plateau(optimizer, reduce_factor, schedule_patience, min_lr, metric_mode):
        #metric_mode = 'min'
        # metric_mode = cfg.metric_agg[-3:]
        # if metric_mode not in ['min', 'max']:
        #     raise ValueError(f"Failed to automatically infer min or max mode "
        #                      f"from cfg.metric_agg='{cfg.metric_agg}'")

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode=metric_mode,
            factor=reduce_factor,
            patience=schedule_patience,
            min_lr=min_lr,
            verbose=True
        )
        if not hasattr(scheduler, 'get_last_lr'):
            # ReduceLROnPlateau doesn't have `get_last_lr` method as of current
            # pytorch1.10; we add it here for consistency with other schedulers.
            def get_last_lr(self):
                """ Return last computed learning rate by current scheduler.
                """
                return self._last_lr

            scheduler.get_last_lr = get_last_lr.__get__(scheduler)
            scheduler._last_lr = [group['lr']
                                  for group in scheduler.optimizer.param_groups]

        return scheduler
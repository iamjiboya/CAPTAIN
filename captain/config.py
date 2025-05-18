import torch

class Config:
    """Configuration class to hold hyperparameters and settings."""
    def __init__(self, defaults):
        for key, value in defaults.items():
            setattr(self, key, value)

hyperparameter_defaults = dict(
    seed=0,
    dataset_name="ms",
    do_train=True,
    load_model="/home/jiboya/captain",
    mask_ratio=0.15,
    epochs=40,
    n_bins=51,
    MVC=False,
    ecs_thres=0.0,
    dab_weight=1.0,
    lr=1e-5,
    batch_size=20,
    layer_size=512,
    nlayers=12,
    nhead=8,
    dropout=0.2,
    schedule_ratio=1,
    save_eval_interval=5,
    fast_transformer=True,
    pre_norm=False,
    amp=True,
    include_zero_gene=False,
    freeze=False,
    DSBN=False,
    use_mod=True,
)
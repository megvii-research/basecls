#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import os

from basecore.config import ConfigDict

__all__ = ["BaseConfig"]

_cfg = dict(
    trainer_name="ClsTrainer",
    hooks_name="DefaultHooks",
    # Weights to start training from
    weights=None,
    # Output directory
    output_dir=None,
    # mini-batch size per device
    batch_size=32,
    # number of classes
    num_classes=1000,
    preprocess=dict(
        img_size=224,
        img_color_space="BGR",
        img_mean=(103.530, 116.280, 123.675),
        img_std=(57.375, 57.12, 58.395),
    ),
    test=dict(
        img_size=224,
        crop_pct=0.875,
    ),
    bn=dict(
        # Precise BN interval in epochs, `float("inf")` to disable, -1 to apply after the last epoch
        precise_every_n_epoch=float("inf"),
        # Number of samples when applying Precise BN
        num_samples_precise=8192,
    ),
    loss=dict(
        # Loss type select from {"BinaryCrossEntropy", "CrossEntropy"}
        name="CrossEntropy",
        # Label smoothing value in 0 to 1 (0 gives no smoothing)
        label_smooth=0.0,
    ),
    augments=dict(
        name="ColorAugment",
        resize=dict(
            scale_range=(0.08, 1.0),
            ratio_range=(3 / 4, 4 / 3),
            # Resize interpolation select from {"bicubic", "bilinear", "lanczos", "nearest"}
            interpolation="bilinear",
        ),
        color_aug=dict(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            lighting=0.1,
        ),
        rand_aug=dict(
            magnitude=9,
            magnitude_std=0.5,
            prob=0.5,
            n_ops=2,
        ),
        rand_erase=dict(
            prob=0.0,
            scale_range=(0.02, 1.0 / 3),
            ratio=0.3,
            mode="const",
            count=1,
        ),
        mixup=dict(
            mixup_alpha=0.0,
            cutmix_alpha=0.0,
            cutmix_minmax=None,
            prob=1.0,
            switch_prob=0.5,
            mode="batch",
            calibrate_cutmix_lambda=True,
            calibrate_mixup_lambda=False,
            permute=False,
        ),
    ),
    data=dict(
        # Dataloader type select from {"FolderLoader", "FakeData"}
        # DPFlow is fast but also costly, so is recommended only for small networks
        name="FolderLoader",
        train_path=None,
        val_path=None,
        # Number of data loader workers per process
        num_workers=6,
    ),
    solver=dict(
        name="DefaultSolver",
        optimizer="sgd",
        # Learning rate ranges from `basic_lr` to `lr_min_factor * basic_lr`
        # according to the lr_schedule, `lr_min_factor` also required by "rel_exp"
        basic_lr=0.0125,
        lr_min_factor=0,
        # Momentum
        momentum=0.9,
        # L2 regularization
        weight_decay=1e-4,
        # Nesterov momentum
        nesterov=False,
        # Coefficients used for computing running averages of gradient
        betas=(0.9, 0.999),
        # Apply adaptive lr to 0.0 weight decay parameters for LARS & LAMB
        always_adapt=False,
        # Maximal number of epochs
        max_epoch=100,
        # Gradually warm up the basic_lr over this number of epochs
        warmup_epochs=5,
        # Start the warm up from basic_lr * warmup_factor
        warmup_factor=0.1,
        # Learning rate schedule select from {"cosine", "exp", "rel_exp", "linear", "step"}
        lr_schedule="step",
        # Learning rate multiplier for "step" or "exp" schedule
        lr_decay_factor=0.1,
        # Stages for "step" schedule (in epochs)
        lr_decay_steps=(30, 60, 90),
        # Grad clip
        grad_clip=dict(
            # Grad clip type select from {None, "norm", "value"}
            name=None,
            # Max norm of grad clip by norm
            max_norm=float("inf"),
            # Lower bound of grad clip by value
            lower=float("-inf"),
            # Upper bound of grad clip by value
            upper=float("inf"),
        ),
        accumulation_steps=1,
    ),
    # Model Exponential Moving Average (EMA)
    model_ema=dict(
        enabled=False,
        # momentum used by model EMA
        momentum=0.9999,
        # Iteration frequency with which to update EMA weights
        update_period=1,
        # start_epoch = None means warmup_epoch usually
        start_epoch=None,
        # pycls style relative update value, if set, momentum is computed by:
        # momentum = 1 - alpha * (total_batch_size / max_epoch * update_period)
        # see: https://github.com/facebookresearch/pycls/pull/138
        alpha=None,
    ),
    # Automatic Mixed Precision (AMP)
    amp=dict(
        enabled=False,
        # by default we scale loss/gradient by a fixed number of 128.
        # when dynamic scale is enabled, we start with a higher scale of 65536,
        # scale is doubled every 2000 iter or halved once inf is detected during training.
        dynamic_scale=False,
    ),
    # PROFILE mode used for speed optimization
    fastrun=False,
    # Dynamic Tensor Rematerialization (DTR) used for memory optimization
    dtr=False,
    # Log interval in iters
    log_every_n_iter=20,
    # Tensorboard interval in iters
    tb_every_n_iter=20,
    # Save checkpoint interval in epochs, `float("inf")` to disable during training
    save_every_n_epoch=10,
    # Evaluation interval in epochs, `float("inf")` to disable during training
    eval_every_n_epoch=1,
    # Enable tracing
    trace=False,
    # Seed
    seed=42,
)


class BaseConfig(ConfigDict):
    def __init__(self, values_or_file=None, **kwargs):
        super().__init__(_cfg)
        self.merge(values_or_file, **kwargs)

    def link_log_dir(self, link_name="log"):
        output_dir = self.output_dir
        if output_dir is None:
            raise ValueError("Output directory is not specified")
        os.makedirs(output_dir, exist_ok=True)

        if os.path.islink(link_name) and os.readlink(link_name) != output_dir:
            os.system("rm " + link_name)
        if not os.path.exists(link_name):
            cmd = f"ln -s {output_dir} {link_name}"
            os.system(cmd)

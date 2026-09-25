"""
Custom learning rate schedulers for Ultra training.
"""
import math


class CurriculumScheduler:
    """
    Learning rate scheduler aligned with observation masking curriculum.

    Schedule:
    - Epoch 0-warmup_end: Constant LR = initial_lr (warmup period, full observations)
    - Epoch warmup_end to decay_end: Linear decay initial_lr → final_lr (matches masking schedule)
    - Epoch decay_end+: Constant LR = final_lr (sparse observations, fine-tuning)

    This aligns with the observation masking curriculum where observations gradually
    become sparser, requiring more careful policy updates.
    """

    def __init__(self, initial_lr, final_lr, warmup_end, decay_end, apply_to_entropy=False, start_entropy_coef=0):
        """
        Args:
            initial_lr: Initial learning rate (e.g., 2e-4)
            final_lr: Final learning rate after decay (e.g., 5e-5)
            warmup_end: Epoch when decay starts (e.g., 500)
            decay_end: Epoch when decay ends (e.g., 5500)
            apply_to_entropy: Whether to also schedule entropy coefficient
            start_entropy_coef: Initial entropy coefficient if apply_to_entropy=True
        """
        # Convert to float to handle both numeric and string inputs from config
        self.initial_lr = float(initial_lr)
        self.final_lr = float(final_lr)
        self.warmup_end = int(warmup_end)
        self.decay_end = int(decay_end)
        self.decay_duration = self.decay_end - self.warmup_end
        self.apply_to_entropy = apply_to_entropy
        self.start_entropy_coef = float(start_entropy_coef)

        assert self.decay_duration > 0, f"decay_end ({self.decay_end}) must be greater than warmup_end ({self.warmup_end})"
        assert self.final_lr <= self.initial_lr, f"final_lr ({self.final_lr}) should be <= initial_lr ({self.initial_lr})"

    def update(self, current_lr, entropy_coef, epoch, kl_dist=None, a_losses=None, scaled=False):
        """
        Update learning rate based on current epoch.

        Args:
            current_lr: Current learning rate value
            entropy_coef: Current entropy coefficient
            epoch: Current training epoch
            kl_dist: KL divergence (unused, for API compatibility)
            a_losses: Actor losses (unused, for API compatibility)
            scaled: Whether values are already scaled (unused, for API compatibility)

        Returns:
            new_lr: Updated learning rate
            new_entropy_coef: Updated entropy coefficient (or unchanged if not scheduled)
        """
        if epoch < self.warmup_end:
            # Phase 1: Warmup - constant high LR
            new_lr = self.initial_lr
        elif epoch < self.decay_end:
            # Phase 2: Decay - linear decrease
            progress = (epoch - self.warmup_end) / self.decay_duration
            new_lr = self.initial_lr + (self.final_lr - self.initial_lr) * progress
        else:
            # Phase 3: Fine-tuning - constant low LR
            new_lr = self.final_lr

        # Optionally schedule entropy coefficient with same pattern
        if self.apply_to_entropy:
            if epoch < self.warmup_end:
                new_entropy_coef = self.start_entropy_coef
            elif epoch < self.decay_end:
                progress = (epoch - self.warmup_end) / self.decay_duration
                new_entropy_coef = self.start_entropy_coef * (1.0 - progress)
            else:
                new_entropy_coef = 0.0
        else:
            new_entropy_coef = entropy_coef

        return new_lr, new_entropy_coef

    def get_lr_info(self, epoch):
        """Get current phase and learning rate for logging."""
        if epoch < self.warmup_end:
            phase = "warmup"
            lr = self.initial_lr
        elif epoch < self.decay_end:
            phase = "decay"
            progress = (epoch - self.warmup_end) / self.decay_duration
            lr = self.initial_lr + (self.final_lr - self.initial_lr) * progress
        else:
            phase = "finetune"
            lr = self.final_lr

        return {
            'phase': phase,
            'lr': lr,
            'progress': min((epoch - self.warmup_end) / self.decay_duration, 1.0) if phase != "warmup" else 0.0
        }


class CosineScheduler:
    """
    Cosine annealing learning rate scheduler with warmup.
    Provides smoother transitions than linear decay.
    """

    def __init__(self, initial_lr, final_lr, warmup_end, decay_end, apply_to_entropy=False, start_entropy_coef=0):
        """
        Args:
            initial_lr: Initial learning rate
            final_lr: Final learning rate after decay
            warmup_end: Epoch when decay starts
            decay_end: Epoch when decay ends
            apply_to_entropy: Whether to also schedule entropy coefficient
            start_entropy_coef: Initial entropy coefficient if apply_to_entropy=True
        """
        # Convert to float to handle both numeric and string inputs from config
        self.initial_lr = float(initial_lr)
        self.final_lr = float(final_lr)
        self.warmup_end = int(warmup_end)
        self.decay_end = int(decay_end)
        self.decay_duration = self.decay_end - self.warmup_end
        self.apply_to_entropy = apply_to_entropy
        self.start_entropy_coef = float(start_entropy_coef)

        assert self.decay_duration > 0, f"decay_end ({self.decay_end}) must be greater than warmup_end ({self.warmup_end})"
        assert self.final_lr <= self.initial_lr, f"final_lr ({self.final_lr}) should be <= initial_lr ({self.initial_lr})"

    def update(self, current_lr, entropy_coef, epoch, kl_dist=None, a_losses=None, scaled=False):
        """Update learning rate with cosine schedule."""
        if epoch < self.warmup_end:
            # Phase 1: Warmup - constant high LR
            new_lr = self.initial_lr
        elif epoch < self.decay_end:
            # Phase 2: Cosine decay
            progress = (epoch - self.warmup_end) / self.decay_duration
            cosine_progress = (1 - math.cos(progress * math.pi)) / 2  # Smooth S-curve
            new_lr = self.initial_lr + (self.final_lr - self.initial_lr) * cosine_progress
        else:
            # Phase 3: Fine-tuning - constant low LR
            new_lr = self.final_lr

        # Optionally schedule entropy coefficient
        if self.apply_to_entropy:
            if epoch < self.warmup_end:
                new_entropy_coef = self.start_entropy_coef
            elif epoch < self.decay_end:
                progress = (epoch - self.warmup_end) / self.decay_duration
                cosine_progress = (1 - math.cos(progress * math.pi)) / 2
                new_entropy_coef = self.start_entropy_coef * (1.0 - cosine_progress)
            else:
                new_entropy_coef = 0.0
        else:
            new_entropy_coef = entropy_coef

        return new_lr, new_entropy_coef

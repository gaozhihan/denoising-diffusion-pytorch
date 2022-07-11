from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from denoising_diffusion_pytorch.datasets.cifar import MyCIFAR10

dataset_dict = {
    "name": "cifar10",
    "torch_dataset": MyCIFAR10(),
    "image_size": 32,
    "train_batch_size": 32,
}

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = dataset_dict["image_size"],
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()

trainer = Trainer(
    diffusion,
    torch_dataset = dataset_dict["torch_dataset"],
    train_batch_size = dataset_dict["train_batch_size"],
    train_lr = 1e-4,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True                        # turn on mixed precision
)

trainer.train()

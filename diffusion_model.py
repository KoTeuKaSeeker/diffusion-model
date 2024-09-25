from comet_ml import Experiment

import math
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from typing import List

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms

import os 


class DeviceManager():
    """
        A special class designed to store information about available devices, 
        and also has methods that can be overridden to run code on the TPU.
    """
    def __init__(self, rank, world_size, master_process, device):
        self.rank = rank
        self.world_size = world_size
        self.master_process = master_process
        self.device = device
    
    def master_print(self, str):
        if self.master_process:
            print(str)
    
    def rendezvous(self, str):
        pass

    def mark_step(self):
        torch.cuda.synchronize()
    
    def optimizer_step(self, optimizer):
        optimizer.step()
    
    def all_reduce(self, tensor):
        return tensor
    
    def save(self, checkpoint, filename):
        torch.save(checkpoint, filename)

    

class TrainParameters():
    """
    Class for storing various training hyperparameters.
    """
    def __init__(self, T, epochs, show_freq, dataset_path, shard_size, batch_size, target_image_size, save_freq, 
                 save_path, model_path, load_model_from_path, sample_output_path):
        self.T = T
        self.epochs = epochs
        self.show_freq = show_freq
        self.dataset_path = dataset_path
        self.shard_size = shard_size
        self.batch_size = batch_size
        self.target_image_size = target_image_size
        self.save_freq = save_freq
        self.save_path = save_path
        self.model_path = model_path
        self.load_model_from_path = load_model_from_path
        self.sample_output_path = sample_output_path

        
class LearningRateScheduler():
    def __init__(self, min_lr, max_lr, warmup_steps_portion):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_steps_portion = warmup_steps_portion
    
    def init(self, train_parameters: TrainParameters, dataset_len):
        count_steps_in_epoch = dataset_len // train_parameters.batch_size
        self.max_steps = count_steps_in_epoch * train_parameters.epochs
        self.warmup_steps = int(self.max_steps * self.warmup_steps_portion)
    
    def get_lr(self, total_iteration):
        # 1) linear warmup for warmup_iters steps
        if total_iteration < self.warmup_steps:
            return self.max_lr * (total_iteration + 1) / self.warmup_steps
        # 2) if it > lr_decay_iters, return min_lerning_rate
        if total_iteration > self.max_steps:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (total_iteration - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return self.min_lr + coeff * (self.max_lr - self.min_lr)

    
class CometManager():
    def __init__(self, api_key: str, project_name: str, workspace: str, device_manager: DeviceManager, use_comet=True):
        if use_comet and device_manager.master_process:
            self.experiment = Experiment(api_key, project_name, workspace)
        else:
            self.experiment = None
        self.device_manager = device_manager
    
    def log_image(self, image_path: str, name: str):
        if self.experiment is not None:
            self.experiment.log_image(image_path, name=name)


class CatDataset(IterableDataset):
    def __init__(self, path: str, shard_size: int, batch_size: int, target_image_size: tuple, 
                 rank: int, world_size: int, device_manager: DeviceManager, full_in_RAM:bool = False, shuffle: bool = False, allows_ex: List[str] = [".jpg", ".png", ".JPEG"]):
        self.full_in_RAM = full_in_RAM
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.device_manager = device_manager

        self.image_paths = [os.path.join(path, name) for name in os.listdir(path) if os.path.splitext(name)[1] in allows_ex]

        self.count_images = len(self.image_paths)
        self.shard_size = self.count_images if full_in_RAM else shard_size
        self.count_shards = self.count_images // shard_size
        
        self.target_image_size = target_image_size

        self.transform = transforms.Compose([
            transforms.Resize(target_image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1)
        ])
        
        self.reverse_transform = transforms.Compose([
            transforms.Lambda(lambda x: (x + 1) / 2),
            transforms.ToPILImage()
        ])

        if shuffle:
            np.random.shuffle(self.image_paths)
        
        if self.full_in_RAM:
            self.shard = self.get_shard(0)


    def get_shard(self, shard_n) -> torch.Tensor:
        """
        Loads a shard of images from the hard drive and then preprocesses it. 
        Shards allow processing huge datasets that do not fit entirely into RAM. 
        Preloading images also affects processing speed.
        """
        assert shard_n < self.count_shards, "shard_n has a greater value than count_shards."
        shard_start_id = shard_n * self.shard_size
        shard_image_paths = self.image_paths[shard_start_id:shard_start_id + self.shard_size]
        
        shard_images = torch.zeros((self.shard_size, 3) + self.target_image_size)
        shard_images.share_memory_()
        if self.rank == 0:
            images = []
            for path in shard_image_paths:
                image = Image.open(path).convert('RGB')
                image = self.transform(image)
                images.append(image)

            torch.stack(images, 0, out=shard_images)
        
        self.device_manager.rendezvous("transfer_data")
        
        return shard_images


    def __iter__(self):
        for shard_id in range(self.count_shards):
            shard_images = self.shard if self.full_in_RAM else self.get_shard(shard_id)
            image_id = self.rank * self.batch_size
            while image_id + self.batch_size <= self.shard_size:
                images_batch = shard_images[image_id:image_id + self.batch_size]
                yield images_batch
                image_id += self.world_size * self.batch_size


    def __len__(self) -> int:
        return self.count_images


class CatDatasetLite(Dataset):
    def __init__(self, path: str, target_image_size: tuple, device_manager: DeviceManager):
        self.image_paths = [os.path.join(path, name) for name in os.listdir(path)]
        self.count_images = len(self.image_paths)
        self.target_image_size = target_image_size
        
        self.transform = transforms.Compose([
            transforms.Resize(target_image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1)
        ])
        
        self.reverse_transform = transforms.Compose([
            transforms.Lambda(lambda x: (x + 1) / 2),
            transforms.ToPILImage()
        ])
        
        shared_images = torch.zeros((self.count_images, 3) + self.target_image_size)
        shared_images.share_memory_()
        if device_manager.master_process:
            images = []
            for path in self.image_paths:
                image = Image.open(path).convert('RGB')
                image = self.transform(image)
                images.append(image)

            torch.stack(images, 0, out=shared_images)
        
        device_manager.rendezvous("load_sync")
        
        self.cat_images = shared_images.detach().clone()
    
    
    def __getitem__(self, index):
        return self.cat_images[index]
    
    
    def __len__(self):
        return self.count_images
        


def clean_folder(path: str, allows_ex: List[str]):
    """
        Filters the folder by allowed extensions.
    """
    for name in os.listdir(path):
        _, ex = os.path.splitext(name)
        if ex not in allows_ex:
            file_path = os.path.join(path, name)
            os.remove(file_path)


class NoiseScheduler(torch.nn.Module):
    """
    Noise generator for the image. It performs a forward pass process on diffuse model.
    """
    def __init__(self, T: int, device_manager: DeviceManager):
        super().__init__()
        self.T = T
        self.device_manager = device_manager

        self.betas = self.linear_beta_schedule(timesteps=T)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
    

    def linear_beta_schedule(self, timesteps, start=0.0001, end=0.02) -> torch.Tensor:
        return torch.linspace(start, end, timesteps, device=self.device_manager.device)


    def get_index_from_list(self, vals: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        batch_size = t.shape[0]
        out = vals.gather(-1, t)
        return out.reshape(batch_size, *((1,)*(len(x_shape) - 1)))


    def forward(self, x_0, t) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x_0, device=self.device_manager.device)
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
    
    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2]
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)

class SinusidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class SimpleUnet(nn.Module):
    def __init__(self):
        super().__init__()
        image_channels = 3
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 3
        time_emb_dim = 32
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], time_emb_dim) for i in range(len(down_channels) - 1)])

        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], time_emb_dim, up=True) for i in range(len(up_channels) - 1)])

        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

        self.reverse_transform = transforms.Compose([
            transforms.Lambda(lambda x: (x + 1) / 2),
            transforms.ToPILImage()
        ])

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        x = self.conv0(x)
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        return self.output(x)
    
    @torch.no_grad()
    def sample_timestep(self, x, t, noise_scheduler: NoiseScheduler):
        t_tensor = torch.tensor([t], device=x.device, dtype=torch.long)
        betas_t = noise_scheduler.get_index_from_list(noise_scheduler.betas, t_tensor, x.shape)
        sqrt_one_minus_alphas_cumprod_t = noise_scheduler.get_index_from_list(noise_scheduler.sqrt_one_minus_alphas_cumprod, t_tensor, x.shape)
        sqrt_recip_alphas_t = noise_scheduler.get_index_from_list(noise_scheduler.sqrt_recip_alphas, t_tensor, x.shape)

        model_mean = sqrt_recip_alphas_t * (x - betas_t * self(x, t_tensor) / sqrt_one_minus_alphas_cumprod_t)
        posterior_variance_t = noise_scheduler.get_index_from_list(noise_scheduler.posterior_variance, t_tensor, x.shape)

        if t == 0:
            return model_mean
        else:
            noise = torch.randn_like(x, device=x.device)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    
    @torch.no_grad()
    def generate_images(self, x, noise_t, noise_scheduler: NoiseScheduler, device_manager: DeviceManager):
        """
        Generates images from a batch of noise images x, 
        reducing its noise from the state t = noise_t (lots of noise) -> t = 0 (no noise).
        """

        for i in range(0, noise_t)[::-1]:
            device_manager.mark_step()
            x = self.sample_timestep(x, i, noise_scheduler)
            # x = torch.clamp(x, -1.0, 1.0)
            device_manager.mark_step()
        
        return  x

@torch.no_grad()
def show_samples(model: SimpleUnet, count_samples: int, image_size: int, device_manager: DeviceManager, noise_scheduler: NoiseScheduler, train_dataset: CatDataset):
    image = torch.randn((1, 3, image_size, image_size), device=device_manager.device)

    fig, axes = plt.subplots(1, count_samples, figsize=(15, 15))
    axes = axes.flatten()
    
    steps_to_show = noise_scheduler.T // count_samples

    for i in range(0, noise_scheduler.T)[::-1]:
        device_manager.mark_step()
        image = model.sample_timestep(image, i, noise_scheduler)
        device_manager.mark_step()
        if i % steps_to_show == 0:
            image_cpu = image[0].clone().cpu()
            if i == 0:
                device_manager.master_print(f"mean: {image_cpu.mean()}, std: {image_cpu.std()}")
            pil_image = train_dataset.reverse_transform(image_cpu)
            ax = axes[i // steps_to_show]
            ax.imshow(pil_image)
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return fig

@torch.no_grad()
def show_noiseless_samples(model, count_samples: tuple, image_size: int, device_manager: DeviceManager, noise_scheduler: NoiseScheduler, train_dataset: CatDataset):
    noises = torch.randn((count_samples[0]*count_samples[1], 3, image_size, image_size), device=device_manager.device)
    
    fig, axes = plt.subplots(count_samples[1], count_samples[0], figsize=(15, 15))
    axes = axes.flatten()
    
    images = model.generate_images(noises, noise_scheduler.T, noise_scheduler, device_manager)
    images = torch.clamp(images, -1, 1)
    images = images.cpu()
    
    for ax, image in zip(axes, images):
        image = train_dataset.reverse_transform(image)
        ax.imshow(image)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return fig


def save_model(filename: str, model: SimpleUnet, optimizer, device_manager: DeviceManager):
    device_manager.rendezvous("start_save")
    device_manager.mark_step()
    
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    
    device_manager.save(checkpoint, filename)
    
    device_manager.rendezvous("end_save")
    device_manager.mark_step()
    
    
def train_loop(model: SimpleUnet, optimizer, noise_scheduler: NoiseScheduler, train_dataset, train_loader, device_manager: DeviceManager, 
               train_parameters: TrainParameters, learning_rate_scheduler: LearningRateScheduler, comet_manager: CometManager):
    total_iteration = 0
    for epoch in range(train_parameters.epochs):
        for step, batch in enumerate(train_loader):
            batch = batch.to(device_manager.device)
            optimizer.zero_grad()

            t = torch.randint(0, noise_scheduler.T, (batch.size(0), ), device=device_manager.device, dtype=torch.long)
            x_noisy, noise = noise_scheduler(batch, t)
            noise_pred = model(x_noisy, t)
            loss = F.l1_loss(noise, noise_pred)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            device_manager.optimizer_step(optimizer) # mark_step is already in use
            
            learning_rate = learning_rate_scheduler.get_lr(total_iteration)
            for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate
            
            device_manager.master_print(f"total_iteration {total_iteration}, epoch {epoch}, step {step} | loss {loss.item()} | lr: {learning_rate:.4e}")
            if total_iteration % train_parameters.show_freq == 0:
                if device_manager.master_process:
                    #figure = show_samples(model, 10, batch.shape[-1], device_manager, noise_scheduler, train_dataset)
                    figure = show_noiseless_samples(model, (5, 5), train_parameters.target_image_size, device_manager, noise_scheduler, train_dataset)
                    plt.figure(figure)
                    plt.savefig(train_parameters.sample_output_path)
                    comet_manager.log_image(train_parameters.sample_output_path, name=f"Step {total_iteration}")
                device_manager.rendezvous("show_samples")
            
            if total_iteration % train_parameters.save_freq == 0:
                device_manager.master_print("Saving model...")
                save_model(train_parameters.save_path, model, optimizer, device_manager)

            total_iteration += 1
    
    save_model(train_parameters.save_path, model, optimizer, device_manager)


def run(device_manager: DeviceManager, train_parameters: TrainParameters, learning_rate_scheduler: LearningRateScheduler, comet_manager: CometManager):
    noise_scheduler = NoiseScheduler(train_parameters.T, device_manager)

    train_dataset = CatDatasetLite(train_parameters.dataset_path, (train_parameters.target_image_size,)*2, device_manager)
    train_sampler = DistributedSampler(train_dataset, num_replicas=device_manager.world_size, rank=device_manager.rank)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_parameters.batch_size, sampler=train_sampler)
    
    model = SimpleUnet()
    model.to(device_manager.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate_scheduler.min_lr, weight_decay=1e-3)
    if train_parameters.load_model_from_path:
        checkpoint = torch.load(train_parameters.model_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    
    learning_rate_scheduler.init(train_parameters, len(train_dataset))        
    
    train_loop(model, optimizer, noise_scheduler, train_dataset, train_loader, device_manager, train_parameters, learning_rate_scheduler, comet_manager)


if __name__ == "__main__":
    torch.manual_seed(234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(234)
    
    # torch.set_float32_matmul_precision('high')

    rank = 0
    world_size = 1
    master_process = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    count_timesteps = 300
    epochs = 5000
    show_freq = 100
    save_freq = 1000
    save_path = r"models\cat-diffusion-checkpoints\model.pt"
    sample_output_path = r"sample_output.png"
    model_path = r"models\cat-diffusion-64px\model.pt"
    load_model_from_path = False
    
    dataset_path = r"data\images"
    shard_size = 1024
    batch_size = 4
    target_image_size = 64
    
    min_learning_rate = 1e-5
    max_learning_rate = 1e-3
    warmup_steps_portion = 0.01
    
    # comet
    comet_api_key = "MB9XnDlVfSVSMuK8PqL6hXvNg"
    comet_project_name = "cat-diffuion"
    comet_workspace = "koteukaseeker"
    use_comet = True
    
    device_manager = DeviceManager(rank, world_size, master_process, device)
    train_parameters = TrainParameters(count_timesteps, epochs, show_freq, dataset_path, shard_size, batch_size, target_image_size, 
                                       save_freq, save_path, model_path, load_model_from_path, sample_output_path)
    learning_rate_scheduler = LearningRateScheduler(min_learning_rate, max_learning_rate, warmup_steps_portion)
    comet_manager = CometManager(comet_api_key, comet_project_name, comet_workspace, device_manager, use_comet=use_comet)
    
    run(device_manager, train_parameters, learning_rate_scheduler, comet_manager)
    
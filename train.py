import torch
import seaborn as sn
import matplotlib.pyplot as plt
import sidechainnet as scn
from PIL import Image

import torch
import torch.nn.functional as F
from torch import optim
from ddpm_proteins import Unet, GaussianDiffusion

# constants

NUM_ITERATIONS = int(2e6)
IMAGE_SIZE = 256
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 8
LEARNING_RATE = 2e-5
SAMPLE_EVERY = 100
SCALE_DISTANCE_BY = 1e2

# experiment tracker

import wandb
wandb.init(project = 'ddpm-proteins-masked')
wandb.run.name = f'proteins of length {IMAGE_SIZE} or less'
wandb.run.save()

# model

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels = 1,
    condition_dim = 1
)

diffusion = GaussianDiffusion(
    model,
    image_size = IMAGE_SIZE,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
)

def cycle(loader, thres = 256):
    while True:
        for data in loader:
            if data.seqs.shape[1] <= thres:
                yield data

data = scn.load(
    casp_version = 12,
    thinning = 30,
    with_pytorch = 'dataloaders',
    batch_size = BATCH_SIZE,
    dynamic_batching = False
)

opt = optim.Adam(diffusion.parameters(), lr = LEARNING_RATE)

train_dl = cycle(data['train'], thres = IMAGE_SIZE)
valid_dl = cycle(data['test'], thres = IMAGE_SIZE)

diffusion = diffusion.cuda()

upper_triangular_mask = torch.ones(IMAGE_SIZE, IMAGE_SIZE).triu_(1).bool().cuda()

for ind in range(NUM_ITERATIONS):
    for _ in range(GRADIENT_ACCUMULATE_EVERY):
        batch = next(train_dl)
        seqs, coords, masks = batch.seqs, batch.crds, batch.msks

        coords = coords.reshape(BATCH_SIZE, -1, 14, 3)
        coords = coords[:, :, 1].cuda() # pick off alpha carbon

        dist = torch.cdist(coords, coords)
        data = dist[:, None, :, :]

        crossed_mask = (masks[:, None, :, None] * masks[:, None, None, :]).cuda()
        data.masked_fill_(~crossed_mask.bool(), 0.)

        remainder = IMAGE_SIZE - data.shape[-1]
        data = F.pad(data, (0, remainder, 0, remainder), value = 0.)
        crossed_mask = F.pad(crossed_mask, (0, remainder, 0, remainder), value = -1.)

        data = (data / SCALE_DISTANCE_BY).clamp(0., 1.)

        data = data * upper_triangular_mask[None, None, :, :]

        loss = diffusion(data, condition_tensor = crossed_mask.float())
        (loss / GRADIENT_ACCUMULATE_EVERY).backward()

    print(loss.item())
    wandb.log({'loss': loss.item()})
    opt.step()
    opt.zero_grad()

    if (ind % SAMPLE_EVERY) == 0:
        batch = next(valid_dl)
        seqs, coords, masks = batch.seqs, batch.crds, batch.msks

        crossed_mask = (masks[:, None, :, None] * masks[:, None, None, :]).cuda()
        remainder = IMAGE_SIZE - crossed_mask.shape[-1]
        crossed_mask = F.pad(crossed_mask, (0, remainder, 0, remainder), value = -1.)[:1].float()

        sampled = diffusion.sample(batch_size = 1, condition_tensor = crossed_mask)[0][0]

        sampled = sampled.clamp(0., 1.) * upper_triangular_mask
        sampled = sampled.cpu().numpy()
        sampled = sampled + sampled.transpose(-1, -2)

        distogram = sn.heatmap(sampled)
        figure = distogram.get_figure()    
        figure.savefig('./validation.tmp.png', dpi = 100)
        plt.clf()

        crossed_mask_img = sn.heatmap(crossed_mask[0][0].cpu().numpy())
        figure = crossed_mask_img.get_figure()
        figure.savefig('./mask.tmp.png', dpi = 100)
        plt.clf()

        img = Image.open('./validation.tmp.png')
        crossed_mask_img = Image.open('./mask.tmp.png')

        wandb.log({'sample': wandb.Image(img), 'mask': wandb.Image(crossed_mask_img)})

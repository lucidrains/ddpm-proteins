import torch
import seaborn as sn
import matplotlib.pyplot as plt
import sidechainnet as scn
from PIL import Image

import torch
import torch.nn.functional as F
from torch import optim
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

# constants

NUM_ITERATIONS = int(2e6)
IMAGE_SIZE = 256
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 8
LEARNING_RATE = 2e-5
VALIDATE_EVERY = 20
SCALE_DISTANCE_BY = 1e2

# experimental tracker

import wandb
wandb.init(project = 'ddpm-proteins')
wandb.run.name = f'proteins of length {IMAGE_SIZE} or less'
wandb.run.save()

# model

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels = 1,
)

diffusion = GaussianDiffusion(
    model,
    channels = 1,
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
dl = cycle(data['train'], thres = IMAGE_SIZE)

diffusion = diffusion.cuda()

upper_triangular_mask = torch.ones(IMAGE_SIZE, IMAGE_SIZE).triu_(1).bool().cuda()

for ind in range(NUM_ITERATIONS):
    for _ in range(GRADIENT_ACCUMULATE_EVERY):
        batch = next(dl)

        seqs, coords, masks = batch.seqs, batch.crds, batch.msks
        coords = coords.reshape(BATCH_SIZE, -1, 14, 3)
        coords = coords[:, :, 1].cuda() # pick off alpha carbon
        masks = masks.cuda()

        dist = torch.cdist(coords, coords)
        dist.masked_fill_((masks[:, :, None] - masks[:, None, :]).bool(), 0.)
        data = dist[:, None, :, :]

        remainder = IMAGE_SIZE - data.shape[-1]
        data = F.pad(data, (0, remainder, 0, remainder), value = 0.)
        data = (data / SCALE_DISTANCE_BY).clamp(0., 1.)

        data = data * upper_triangular_mask[None, None, :, :]

        loss = diffusion(data)
        (loss / GRADIENT_ACCUMULATE_EVERY).backward()

    print(loss.item())
    wandb.log({'loss': loss.item()})
    opt.step()
    opt.zero_grad()

    if (ind % VALIDATE_EVERY) == 0:
        sampled = diffusion.sample(batch_size = 1)[0][0]

        sampled = sampled.clamp(0., 1.) * upper_triangular_mask
        sampled = sampled.cpu().numpy()
        sampled = sampled + sampled.transpose(-1, -2)

        distogram = sn.heatmap(sampled)
        figure = distogram.get_figure()    
        figure.savefig('./validation.tmp.png', dpi = 100)
        plt.clf()

        img = Image.open('./validation.tmp.png')
        wandb.log({'sample': wandb.Image(img)})

import torch
import sidechainnet as scn

from PIL import Image
from random import randrange

import torch
import torch.nn.functional as F
from torch import optim

from ddpm_proteins import Unet, GaussianDiffusion
from ddpm_proteins.utils import save_heatmap, broadcat, get_msa_attention_embeddings, symmetrize, get_msa_transformer, pad_image_to

from einops import rearrange

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
wandb.init(project = 'ddpm-proteins')
wandb.run.name = f'proteins of length {IMAGE_SIZE} or less'
wandb.run.save()

# model

model = Unet(
    dim = 32,
    dim_mults = (1, 2, 4, 8),
    channels = 1,
    condition_dim = 1 + 144  # mask (1) + attention embedding size (144)
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

model, batch_converter = get_msa_transformer()
model = model.cuda(1) # put msa transformer on cuda device 1

opt = optim.Adam(diffusion.parameters(), lr = LEARNING_RATE)

train_dl = cycle(data['train'], thres = IMAGE_SIZE)
valid_dl = cycle(data['test'], thres = IMAGE_SIZE)

diffusion = diffusion.cuda()

upper_triangular_mask = torch.ones(IMAGE_SIZE, IMAGE_SIZE).triu_(1).bool().cuda()

for ind in range(NUM_ITERATIONS):
    for _ in range(GRADIENT_ACCUMULATE_EVERY):
        batch = next(train_dl)
        ids, seqs, coords, masks = batch.pids, batch.seqs, batch.crds, batch.msks
        seqs = seqs.argmax(dim = -1)

        coords = coords.reshape(BATCH_SIZE, -1, 14, 3)
        coords = coords[:, :, 1].cuda() # pick off alpha carbon

        dist = torch.cdist(coords, coords)
        data = dist[:, None, :, :]

        crossed_mask = (masks[:, None, :, None] * masks[:, None, None, :]).cuda()
        data.masked_fill_(~crossed_mask.bool(), 0.)

        data = pad_image_to(data, IMAGE_SIZE, value = 0.)
        crossed_mask = pad_image_to(crossed_mask, IMAGE_SIZE, value = -1.)

        data = (data / SCALE_DISTANCE_BY).clamp(0., 1.)

        data = data * upper_triangular_mask[None, None, :, :]

        msa_attention_embeds = get_msa_attention_embeddings(model, batch_converter, seqs, ids)
        msa_attention_embeds = pad_image_to(msa_attention_embeds, IMAGE_SIZE)

        condition_tensor = broadcat((msa_attention_embeds.cuda(0), crossed_mask.float()), dim = 1)

        loss = diffusion(data, condition_tensor = condition_tensor)
        (loss / GRADIENT_ACCUMULATE_EVERY).backward()

    print(loss.item())
    wandb.log({'loss': loss.item()})
    opt.step()
    opt.zero_grad()

    if (ind % SAMPLE_EVERY) == 0:
        batch = next(valid_dl)
        ids, seqs, coords, masks = batch.pids, batch.seqs, batch.crds, batch.msks
        seqs = seqs.argmax(dim = -1)

        coords = coords.reshape(BATCH_SIZE, -1, 14, 3)
        coords = coords[:, :, 1].cuda() # pick off alpha carbon

        dist = torch.cdist(coords, coords)
        data = dist[:, None, :, :]

        crossed_mask = (masks[:, None, :, None] * masks[:, None, None, :]).cuda()
        data.masked_fill_(~crossed_mask.bool(), 0.)

        data = pad_image_to(data, IMAGE_SIZE, value = 0.)
        valid_data = (data / SCALE_DISTANCE_BY).clamp(0., 1.)

        crossed_mask = pad_image_to(crossed_mask, IMAGE_SIZE, value = -1.)[:1].float()

        msa_attention_embeds = get_msa_attention_embeddings(model, batch_converter, seqs[:1], ids[:1])
        msa_attention_embeds = pad_image_to(msa_attention_embeds, IMAGE_SIZE)

        condition_tensor = broadcat((msa_attention_embeds.cuda(0), crossed_mask.float()), dim = 1)

        sampled = diffusion.sample(batch_size = 1, condition_tensor = condition_tensor)[0][0]

        sampled = sampled.clamp(0., 1.) * upper_triangular_mask
        sampled = symmetrize(sampled)

        img              = save_heatmap(sampled, './validation.tmp.png', dpi = 100, return_image = True)
        crossed_mask_img = save_heatmap(crossed_mask[0][0], './mask.tmp.png', dpi = 100, return_image = True)
        truth_img        = save_heatmap(valid_data[0][0], './truth.tmp.png', dpi = 100, return_image = True)

        wandb.log({'sample': wandb.Image(img), 'mask': wandb.Image(crossed_mask_img), 'truth': wandb.Image(truth_img)})

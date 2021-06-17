import os
from PIL import Image
import seaborn as sn
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from sidechainnet.utils.sequence import ProteinVocabulary
from einops import rearrange

# general functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def broadcat(tensors, dim = -1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]

    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))

    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim = dim)

# singleton msa transformer

msa_instances = None

def get_msa_transformer():
    global msa_instances
    if not exists(msa_instances):
        msa_model, alphabet = torch.hub.load("facebookresearch/esm", "esm_msa1_t12_100M_UR50S")
        batch_converter = alphabet.get_batch_converter()
        return msa_model, batch_converter
    return msa_instances

# MSA embedding related functions

VOCAB = ProteinVocabulary()

def ids_to_aa_str(x):
    assert isinstance(x, list), 'input must be a list'
    id2aa = VOCAB._int2char
    is_char = lambda c: isinstance(c, str) and len(c) == 1
    out = []

    for el in x:
        if isinstance(el, list):
            out.append(ids_to_aa_str(el))
        elif isinstance(el, int):
            out.append(id2aa[el])
        else:
            raise TypeError('type must be either list or character')

    if all(map(is_char, out)):
        return ''.join(out)

    return out

def aa_str_to_embed_input(x):
    assert isinstance(x, list), 'input must be a list'
    out = []

    for el in x:
        if isinstance(el, list):
            out.append(aa_str_to_embed_input(el))
        elif isinstance(el, str):
            out.append((None, el))
        else:
            raise TypeError('type must be either list or string')

    return out

def apc(x):
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)
    avg = a1 * a2
    avg.div_(a12)
    normalized = x - avg
    return normalized

def symmetrize(x):
    return x + x.transpose(-1, -2)

def pad_image_to(tensor, size, value = 0.):
    remainder = size - tensor.shape[-1]
    tensor = F.pad(tensor, (0, remainder, 0, remainder), value = value)
    return tensor

# getting a single MSA attention embedding, with caching

CACHE_PATH = default(os.getenv('CACHE_PATH'), os.path.expanduser('~/.cache.ddpm-proteins'))
FETCH_FROM_CACHE = not exists(os.getenv('CLEAR_CACHE'))

os.makedirs(CACHE_PATH, exist_ok = True)

@torch.no_grad()
def get_msa_attention_embedding(
    model,
    batch_converter,
    aa_str,
    id,
    fetch_msas_fn = lambda t: [],
    cache = True
):
    device = next(model.parameters()).device

    cache_full_path = os.path.join(CACHE_PATH, f'{id}.pt')
    if cache and FETCH_FROM_CACHE and os.path.exists(cache_full_path):
        try:
            loaded = torch.load(cache_full_path).to(device)
        except:
            loaded = None

        if exists(loaded):
            return loaded

    msas = default(fetch_msas_fn(aa_str), [])
    seq_with_msas = [aa_str, *msas]

    embed_inputs = aa_str_to_embed_input(seq_with_msas)
    _, _, msa_batch_tokens = batch_converter(embed_inputs)

    results = model(msa_batch_tokens.to(device), need_head_weights = True)

    attentions = results['row_attentions']
    attentions = attentions[..., 1:, 1:]
    attentions = rearrange(attentions, 'b l h m n -> b (l h) m n')
    attentions = apc(symmetrize(attentions))

    if cache:
        print(f'caching to {cache_full_path}')
        torch.save(attentions, cache_full_path)

    return attentions

def get_msa_attention_embeddings(
    model,
    batch_converter,
    seqs,
    ids,
    fetch_msas_fn = lambda t: [],
    cache = True
):
    n = seqs.shape[1]
    seqs = rearrange(seqs, 'b n -> b () n')
    aa_strs = ids_to_aa_str(seqs.cpu().tolist())
    embeds_list = [get_msa_attention_embedding(model, batch_converter, aa, seq_id, cache = cache) for aa, seq_id in zip(aa_strs, ids)]
    embeds_list = [pad_image_to(embed, n) for embed in embeds_list]
    embeds = torch.cat(embeds_list, dim = 0)
    return embeds

# training utils

def cycle(loader, thres = 256):
    while True:
        for data in loader:
            if data.seqs.shape[1] <= thres:
                yield data

def save_heatmap(tensor, filepath, dpi = 200, return_image = False):
    heatmap = sn.heatmap(tensor.cpu().numpy())
    figure = heatmap.get_figure()    
    figure.savefig(filepath, dpi = dpi)
    plt.clf()

    if not return_image:
        return
    return Image.open(filepath)

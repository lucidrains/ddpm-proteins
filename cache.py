from tqdm import tqdm
import sidechainnet as scn
from ddpm_proteins.utils import get_msa_attention_embeddings, get_msa_transformer

# sidechainnet data

data = scn.load(
    casp_version = 12,
    thinning = 30,
    with_pytorch = 'dataloaders',
    batch_size = 1,
    dynamic_batching = False
)

# constants

LENGTH_THRES = 256

# function for fetching MSAs, fill-in depending on your setup

def fetch_msas_fn(aa_str):
    """
    given a protein as amino acid strings
    fill in a function that returns the MSAs, also as amino acid strings, as a list
    as default, return nothing, and just pass the sequence into MSA Transformer by itself
    """
    return []

# caching loop

model, batch_converter = get_msa_transformer()

for batch in tqdm(data['train']):
    if batch.seqs.shape[1] > LENGTH_THRES:
        continue

    pids = batch.pids
    seqs = batch.seqs.argmax(dim = -1)

    _ = get_msa_attention_embeddings(
        model,
        batch_converter,
        seqs,
        batch.pids,
        fetch_msas_fn
    )

print('caching complete')

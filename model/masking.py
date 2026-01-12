import torch
import torch.nn.attention.flex_attention as fa

def lengths_to_doc_ids(lengths: torch.Tensor) -> torch.Tensor:
    return torch.arange(
        lengths.shape[0],
        dtype=lengths.dtype,
        device=lengths.device,
    ).repeat_interleave(lengths)

def lengths_to_positions(lengths: torch.Tensor) -> torch.Tensor:
    construct_args = { 'device': lengths.device, 'dtype': lengths.dtype }
    cumsum = lengths.cumsum(dim=0)
    total_len = cumsum[-1].item()
    excl_cumsum = torch.concat([
        torch.tensor([0], **construct_args),
        cumsum[:-1],
    ])
    return torch.arange(total_len, **construct_args) - excl_cumsum.repeat_interleave(lengths)

# note: `doc_ids` is in latents; if `bytes_per_latent` is 4, then each
# position in `doc_ids` refers to 4 bytes.
def create_fa_doc_mask(
    doc_ids: torch.Tensor,
    bytes_per_latent: int = 0,
    q_is_bytes: bool = False,
    kv_is_bytes: bool = False,
    additional_mask = None,
    compile = False,
) -> fa.BlockMask:
    latent_len = doc_ids.shape[0]
    q_len = latent_len if not q_is_bytes else latent_len * bytes_per_latent
    kv_len = latent_len if not kv_is_bytes else latent_len * bytes_per_latent

    def mask_mod(_b, _h, q_idx, kv_idx) -> torch.Tensor:
        if q_is_bytes:
            q_idx = q_idx // bytes_per_latent
        if kv_is_bytes:
            kv_idx = kv_idx // bytes_per_latent

        out = doc_ids[q_idx] == doc_ids[kv_idx]
        if additional_mask is not None:
            out = out & additional_mask(q_idx, kv_idx)
        return out

    return fa.create_block_mask(
        mask_mod, None, None, q_len, kv_len,
        device=doc_ids.device, _compile=compile
    )

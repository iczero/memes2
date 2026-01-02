- using 3.14+gil for now, when triton supports nogil, maybe switch to freethreaded again
- RMSNorm/LayerNorm equivalence: <https://arxiv.org/pdf/2305.14858>
- hyperconnections: <https://arxiv.org/pdf/2409.19606>
- manifold-constrained hyperconnections: <https://arxiv.org/pdf/2512.24880>

```py
# strided rope is an experiment
# byte-level: 0 1 2 3   4 5 6 7
#     latent:   1         5
stride_start = (stride - 1) // 2
cos_cached = self.cos_cached[stride_start::stride, :]
sin_cached = self.sin_cached[stride_start::stride, :]
```

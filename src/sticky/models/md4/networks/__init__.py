"""Neural network components used by MD4.

Includes:
- transformer: LLAMA2-ish Transformer blocks with (optional) adaLN conditioning
- unet: UNet-like residual stack used in MD4 image classifier
- dit / uvit: optional DiT-in-UNet mid-block (used when n_dit_layers>0)

"""

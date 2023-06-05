---
tags:
- pytorch
- diffusers
- unconditional-image-generation
- diffusion-models-class
---

# Multi-epoch (100) unconditional image generation model finetuned with modern architecture data

Multi-epoch (100) unconditional image generation model finetuned with modern architecture data

## Usage

```python
from diffusers import DDPMPipeline

pipeline = DDPMPipeline.from_pretrained('benlehrburger/100Epoch-LSUN-church-finetuned-modern-model')
image = pipeline().images[0]
image
```

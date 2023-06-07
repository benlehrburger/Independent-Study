---
license: creativeml-openrail-m
tags:
- pytorch
- diffusers
- stable-diffusion
- text-to-image
---

# DreamBooth model with special identifier for my pet rabbit, Nala.

This is a Stable Diffusion model fine-tuned with DreamBooth. It can be used by modifying the `instance_prompt`: **a photo of Nala the bunny in the Acropolis**

## Description


This is a Stable Diffusion model fine-tuned on images of my pet rabbit, Nala.


## Usage

```python
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_pretrained('benlehrburger/animal-image-generation')
image = pipeline().images[0]
image
```

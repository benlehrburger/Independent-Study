from huggingface_hub import HfApi, ModelCard, create_repo, get_full_repo_name


# <-- GLOBAL PARAMS -->


start_model = "openai/clip-vit-base-patch32"
dataset_name = "benlehrburger/dreambooth-animal"
model_name = "animal-image-generation-deep"
local_folder_name = "dreambooth-animal-deep"


# <-- PUSH TO HUB -->


description = "DreamBooth model with special identifier for my pet rabbit, Nala. (but deeper)"
hub_model_id = get_full_repo_name(model_name)
create_repo(hub_model_id)
api = HfApi()
api.upload_folder(
    folder_path=f"{local_folder_name}/scheduler", path_in_repo="", repo_id=hub_model_id
)
api.upload_folder(
    folder_path=f"{local_folder_name}/unet", path_in_repo="", repo_id=hub_model_id
)
api.upload_file(
    path_or_fileobj=f"{local_folder_name}/model_index.json",
    path_in_repo="model_index.json",
    repo_id=hub_model_id,
)

# Add a model card
content = f"""
---
tags:
- pytorch
- diffusers
- unconditional-image-generation
- diffusion-models-class
---

# DreamBooth model with special identifier for my pet rabbit, Nala. (but deeper)

This is a Stable Diffusion model fine-tuned with DreamBooth. It can be used by modifying the instance_prompt: a photo of Nala the bunny in the Acropolis

## Usage

```python
from diffusers import DDPMPipeline

pipeline = DDPMPipeline.from_pretrained('{hub_model_id}')
image = pipeline().images[0]
image
```
"""

card = ModelCard(content)
card.push_to_hub(hub_model_id)

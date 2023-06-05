from huggingface_hub import ModelCard, create_repo, get_full_repo_name, Repository


# <-- GLOBAL PARAMS -->


start_model = "distilbert-base-uncased"
dataset_name = "benlehrburger/college-text-corpus"
model_name = "masked-language-finetuned-model"
local_folder_name = "distilbert-finetuned-personal-writer"


# <-- PUSH TO HUB -->


description = "Masked language autocomplete tool developed by finetuning distilbert with 50K writing samples from my Google Drive (10 epochs)"
repo_name = get_full_repo_name(model_name)
output_dir = model_name
repo = Repository(output_dir, clone_from=repo_name)


# Add a model card
content = f"""
---
tags:
- pytorch
- masked-language-models
- autocompletion
- distilbert
---

# Masked language autocomplete tool developed by finetuning distilbert with 50K writing samples from my Google Drive (10 epochs)

Masked language autocomplete tool developed by finetuning distilbert with 50K writing samples from my Google Drive (10 epochs)

## Usage

```python
from transformers import pipeline

mask_filler = pipeline(
    "fill-mask", model='{hub_model_id}'
)

preds = mask_filler("YOUR_TEXT_HERE")
```
"""

card = ModelCard(content)
card.push_to_hub(hub_model_id)

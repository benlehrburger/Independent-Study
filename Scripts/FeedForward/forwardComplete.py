from transformers import pipeline
import torch
import random


# <-- GLOBAL PARAMS -->

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mask_filler = pipeline(
    "fill-mask", model="benlehrburger/masked-language-finetuned-model"
)

#prompts = ["Cognitive [MASK]", "I'm looking for [MASK]", "I go to school in [MASK]", "One day we will [MASK]"]

#prompts = ["I was born in [MASK]", "I was excited to see [MASK]", "Ben [MASK]", "We are going to be destroyed by [MASK]", "I study [MASK]"]

prompts = ["I am interested in [MASK]", "Can you give me a job in [MASK]", "I believe in [MASK]"]

out = "textOut.txt"
with open(out, 'a') as f:
    for prompt in prompts:

        preds = mask_filler(prompt)
        chicken_dinner = ""
        max_score = 0
        for pred in preds:
            if pred["score"] > max_score:
                chicken_dinner = pred["token_str"]
                max_score = pred["score"]
        f.write(f"\n{prompt} ")
        f.write(f">>> {pred['sequence']}")

print("wrote to file")

#for w in range(0, number_of_words):
#    prompt += " "
#    prompt_mask = prompt + "[MASK]"
#    preds = mask_filler(prompt_mask)
#    chicken_dinner = ""
#    max_score = 0
#    for pred in preds:
#        if pred["score"] > max_score:
#            chicken_dinner = pred["token_str"]
#            max_score = pred["score"]
#
#    prompt += chicken_dinner
    

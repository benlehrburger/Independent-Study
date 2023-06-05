from diffusers import DDPMScheduler, PNDMScheduler, StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
import torch
from PIL import Image


# <-- GLOBAL PARAMS -->


prompt = "Nala the bunny enjoying jumping in the air on a trampoline"
#prompt = "Nala the bunny wearing a space helmet while floating in deep purple gassy space"
num_imgs = 4
guidance_scale = 8

model_name = "animal-image-generation-deep"


# <-- MODEL LOADING -->


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipe = StableDiffusionPipeline.from_pretrained(
    "benlehrburger/animal-image-generation-deep",
    torch_dtype=torch.float16,
).to(device)


# <-- IMAGE GENERATION -->


all_images = []
img_count = 1
for _ in range(num_imgs):
    print(f"generating image no. {img_count}")
    images = pipe(prompt, guidance_scale=guidance_scale).images
    images[0].save(f"nala{img_count}.png")
    print('successfully saved!')
    img_count += 1
    
print("done!")

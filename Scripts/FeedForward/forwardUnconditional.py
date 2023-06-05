from diffusers import DDPMPipeline, DDIMScheduler, PNDMScheduler, StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from torchvision.utils import save_image
from tqdm.auto import tqdm
from PIL import Image
import torch


# <-- GLOBAL PARAMS -->


num_samples = 8
theme = "design"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# <-- MODEL LOADING -->


# Load the pretrained pipeline
pipeline_name = "benlehrburger/100Epoch-LSUN-church-finetuned-modern-model"
image_pipe = DDPMPipeline.from_pretrained(pipeline_name).to(device)

# Sample some images with a DDIM Scheduler over 40 steps
scheduler = DDIMScheduler.from_pretrained(pipeline_name)
scheduler.set_timesteps(num_inference_steps=40)

# Random starting point (batch of 8 images)
x = torch.randn(num_samples, 3, 256, 256).to(device)

for i, t in tqdm(enumerate(scheduler.timesteps)):
    model_input = scheduler.scale_model_input(x, t)
    with torch.no_grad():
        noise_pred = image_pipe.unet(model_input, t)["sample"]
    x = scheduler.step(noise_pred, t, x).prev_sample

for itr in range(num_samples):
    img = x[itr]
    save_image(img, f"{theme}{itr}.png")
    
    
print("done!")

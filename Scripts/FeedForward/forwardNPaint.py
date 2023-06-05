from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import random
from PIL import Image
import torch


# <-- GLOBAL PARAMS -->


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
theme = "img2img"
samples = 8
#prompts = ["Ultra-modern office building, high resolution", "modern glass mansion", "sleek modern bungalo", "sleek and minimalistic house with a monochromatic color scheme and clean lines", "modern house with a swimming pool that seamlessly blends with the surrounding architecture, creating a luxurious oasis", "modern house with a beautifully landscaped backyard, featuring a pergola, outdoor kitchen, and a fire pit for entertaining guests", "modern house with an eco-friendly approach, incorporating solar panels, rainwater harvesting systems, and sustainable materials", "modern house with a spacious home office, featuring large windows, a sleek desk setup, and plenty of natural light to inspire productivity", "modern house with floor-to-ceiling windows that overlook a breathtaking natural landscape", "modern house with a unique architectural feature, such as a cantilevered design or a striking asymmetrical facade"]

#prompts = ["modern office building with a sleek glass facade, reflecting the surrounding cityscape and creating a sense of transparency and openness", "state-of-the-art office building with a dynamic and futuristic architectural design, featuring unique angles and geometric patterns", "modern office building with a green rooftop garden, providing employees with a peaceful outdoor space and contributing to sustainability efforts", "modern office building with a spacious atrium at its center, filled with natural light, greenery, and collaborative seating areas to foster creativity and interaction", "modern office building with cutting-edge amenities, such as a fitness center, meditation rooms, and a gourmet cafeteria, promoting employee well-being and work-life balance", "office building designed for sustainability, incorporating features like solar panels, rainwater harvesting systems, and efficient insulation to minimize environmental impact", "modern office building with an emphasis on collaborative spaces, showcasing open lounges, breakout areas, and vibrant meeting rooms to encourage teamwork and innovation", "office building that prioritizes employee productivity and creativity, featuring abundant natural light, comfortable ergonomic furniture, and inspiring artwork throughout the space"]

prompts = ["modern church renovation that blends contemporary design elements with the original architectural features, preserving the historical charm while introducing a fresh and updated aesthetic", "modern church with a stunning glass entrance, creating a welcoming and transparent space that connects the interior with the surrounding community", "modern church renovation that incorporates sustainable and eco-friendly features, such as solar panels, energy-efficient lighting, and rainwater harvesting systems", "modern church with a versatile multipurpose space, allowing for various community activities, events, and gatherings beyond religious services", "modern church renovation that embraces technology, featuring state-of-the-art audiovisual equipment, digital displays, and interactive multimedia installations to enhance worship experiences", "modern church with a contemporary worship hall, utilizing modern materials, lighting, and acoustics to create a comfortable and immersive environment for congregants", "modern church that pays homage to its religious symbolism through carefully crafted contemporary stained glass windows, capturing the play of light and color", "modern church renovation that integrates sustainable landscaping and outdoor gathering spaces, allowing congregants to connect with nature and enjoy outdoor ceremonies and events"]


# <-- MODEL LOADING -->


# Load the pretrained pipeline
pipeline_name = "runwayml/stable-diffusion-v1-5"
image_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(pipeline_name, torch_dtype=torch.float16).to(device)

image_path = "./drab4.jpg"
init_image = Image.open(image_path)

def round_size(x):
    return ((x + 7) & (-8))

width, height = init_image.size
newsize = (round_size(width), round_size(height))
init_image = init_image.resize(newsize)

convert_tensor = transforms.ToTensor()
init_image = convert_tensor(init_image)
#init_image = F.interpolate(init_image)

for itr in range(samples):

#    random_strength = random.uniform(0.6, 0.9)
#    random_guidance = random.uniform(6, 9)
    
    # Apply Img2Img
    result_image = image_pipe(
        prompt = prompts[itr],
        image = init_image, # The starting image
        strength = 0.7, # 0 for no change, 1.0 for max strength
        guidance_scale = 7.5
    ).images[0]

    print(type(result_image))

    #save_image(result_image, f"{theme}{itr}.png")
    result_image = result_image.save(f"{theme}{itr}.png")


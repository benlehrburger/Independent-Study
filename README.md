# Generative AI: Empowering Creatives with Specialized Image and Language Models

I'll cut to the chase. Generative AI is cool but no, I don't think it's going destroy humanity.

I wanted to find some interesting applications of generative AI for my independent study in Cognitive Science @ Dartmouth College. Formally, my focus  area is "How can understanding the brain help us build better tools?" I have learned that Generative AI is a powerful application of brain-based software that affords incredible opportunities for creating tools that improve the human experience. Whether it's neurons or nodes, the emergent properties resulting from millions of finetuned dials never ceases to amaze me.

## Table of Contents

1. [Outputs](#outputs)  
1-1. [Text-Guided Image-to-Image Generation for Modern Architecture Design](##text-guided-image-to-image-generation-for-modern-architecture-design)  
1-2. [Masked Language Models for Personalized Autocompletion](##masked-language-models-for-personalized-autocompletion)  
2. [Training](#training)  
2-1. [Data](###data)  
2.2. [Models](###models)  
2-3. [Computing](###computing)  
2-4. [Scripts](###scripts)
3. [Intermediary Models](#intermediary-models)  
4. [Works Cited](#works-cited)  
5. [Next Steps](#next-steps)  

# Outputs

My best findings comprise the following tools. I cast a wide net into the pool of generative AI and only pursued what I perceived to be the most promising routes. Generative image and language tools have seen remarkable growth over the past couple years and are now open-source, generalizable, and customizable. Check out what I was able to create below.

## Text-Guided Image-to-Image Generation for Modern Architecture Design

<p align="center">
    <img src="Outputs/GenerativeArchitecture/Img2Img/House/Input.jpg" alt="Input Image" width="45%" />
    <img src="./Outputs/GenerativeArchitecture/Img2Img/House/Out5.png" alt="Output Image" width="45%" />
</p>

<p align="center">
    <em>Prompt = "modern house with a spacious home office, featuring large windows, a sleek desk setup, and plenty of natural light to inspire productivity"</em>
</p>

This text-guided image-to-image model takes as input a prompt and an image and returns an ammended version of the image based on the prompt. The model is built on top of [Stable Diffusion 2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) and finetuned for 75 epochs on a [custom web-scraped modern architecture dataset](https://huggingface.co/datasets/benlehrburger/modern-architecture). My vision for this model was to serve as the foundations of a tool to assist architects and designers in thinking outside of any box they may be confined to.

<p align="center">
    <img src="Outputs/GenerativeArchitecture/Img2Img/Office/Example-1/Input.jpg" alt="Input Image" width="45%" />
    <img src="Outputs/GenerativeArchitecture/Img2Img/Office/Example-1/Out1.png" alt="Output Image" width="45%" />
</p>

<p align="center">
    <em>Prompt = "office building designed for sustainability, incorporating features like solar panels, rainwater harvesting systems, and efficient insulation to minimize environmental impact"</em>
</p>

To be a fully deployable tool, the model should be further finetuned on organic architecture (think [Zaha Hadid](https://www.zaha-hadid.com/)). Right now it produces a lot of "classically" modern results that are not as outlandish as the [Parametricism Era](https://en.wikipedia.org/wiki/Parametricism) of architecture – based on CAD modeling and algorithmic design – permits. It also suffers from the classic image generation pitfall of warping and bending reality when you take a closer look. However, it is an interesting first step towards new architectural possibilities.

<p align="center">
    <img src="Outputs/GenerativeArchitecture/Img2Img/Church/Input.jpg" alt="Input Image" width="45%" />
    <img src="Outputs/GenerativeArchitecture/Img2Img/Church/Out1.png" alt="Output Image" width="45%" />
</p>

<p align="center">
    <em>Prompt = "modern church that pays homage to its religious symbolism through carefully crafted contemporary stained glass windows, capturing the play of light and color"</em>
</p>

Yes, those prompts were generated with ChatGPT. I'm really leaning into this whole thing.

## Masked Language Models for Personalized Autocompletion

I trained a masked language model to autocomplete my next word based on my undergraduate writings. The vision is something like a personalized Grammarly but with control over your data. I compiled a [training corpus](https://huggingface.co/datasets/benlehrburger/college-text-corpus) of over 3000 lines of writing samples from essays I had written during my time in college, which I used to finetune [DistilBERT](https://huggingface.co/distilbert-base-uncased) for 50 epochs. The "masked" in masked language model represents the word to be predicted, like so:

>Cognitive [MASK] >>> cognitive neuroscience

The model does well at predicting words in a coherent way.

>I'm looking for [MASK] >>> i'm looking for answers  
>I believe in [MASK] >>> i believe in truth

But it doesn't exactly understand who I am.

>I go to school in [MASK] >>> i go to school in bangkok  
>Ben [MASK] >>> ben!

I think if I were to give the tool more of a capacity for "long-term memory" moving forwards and consistently train it, it would really start to become a helpful, personalized writing aide. But right now it's just kind of weirdly pessimistic:

>One day we will [MASK] >>> one day we will die

# Training

### Data

• [Modern Architecture](https://huggingface.co/datasets/benlehrburger/modern-architecture)  
• [College Text Corpus](https://huggingface.co/datasets/benlehrburger/college-text-corpus)  

### Models

• [Generative Architecture](./Models/ImageGeneration)  
• [Masked Language Model](./Models/LanguageModels/AutoComplete)  

### Computing

I ran most of my computations on Dartmouth's [Discovery Cluster](https://rc.dartmouth.edu/index.php/discovery-overview/) GPUs, in conjunction with [Dartmouth Research Computing](https://rc.dartmouth.edu/). The remote cluster drastically sped up training and feed-forward runs, by about 10^2 the speed my CPU was running at. Special thank you to Kunal Jha, a D'24 ML wizard, who helped me get set up with the cluster. I intend to add my documentation notes to this repository in coming weeks (which I found to be very helpful). Dartmouth alum Jin Hyun Cheong also wrote a detailed [documentation](https://jinhyuncheong.com/jekyll/update/2016/07/24/How-to-use-the-Discovery-cluster.html) that I would recommend.

### Scripts

• [Train Generative Architecture](./Scripts/Train/archGenDepth.py)  
• [Train Autocomplete](./Scripts/Train/archGenDepth.py)  
• [Feed-Forward Architecture](./Scripts/FeedForward/forwardNPaint.py)  
• [Feed-Forward Autocomplete](./Scripts/FeedForward/forwardComplete.py)  

# Intermediary Models

### Dreambooth Unconditional Pet Diffusion

<p align="center">
    <img src="./Outputs/DreamBooth/Beach2.png" alt="Nala on Beach" width="45%" />
    <img src="./Outputs/DreamBooth/Beach1.png" alt="Nala on Beach" width="45%" />
</p>

<p align="center">
    <em>Prompt = "Nala the bunny on the beach"</em>
</p>

This [model](./Models/ImageGeneration/DreamBooth) was made using Dreambooth, which is a technique to teach new concepts to Stable Diffusion with very little training data. My [dataset](https://huggingface.co/datasets/benlehrburger/dreambooth-animal) for this model consists of just 17 images of my pet bunny, Nala. While she may seem slightly demented is those generated images above, it's actually not that far off from what she actually looks like:

<p align="center">
    <img src="https://datasets-server.huggingface.co/assets/benlehrburger/dreambooth-animal/--/benlehrburger--dreambooth-animal/train/4/image/image.jpg" alt="Nala Smush" width="45%" />
    <img src="https://datasets-server.huggingface.co/assets/benlehrburger/dreambooth-animal/--/benlehrburger--dreambooth-animal/train/10/image/image.jpg" alt="Nala Posing" width="45%" />
</p>

This technique was fun because it's a lot of bang for your buck. Five minutes to create a dataset, 2 minutes on a GPU, and it's ready to go. That said, it certainly didn't work all of the time. Just look at the difference in these two samples generated from the same batch:

<p align="center">
    <img src=".Outputs/DreamBooth/Good-Ex.png" alt="Nala at Acropolis" width="45%" />
    <img src="./Outputs/DreamBooth/Bad-Ex.png" alt="Nala at Acropolis" width="45%" />
</p>

<p align="center">
    <em>Prompt = "Nala the bunny at the Acropolis"</em>
</p>

### Stable Diffusion Unconditional Church Generation

### Naive Chatbot Trained on Text Message Data

# Works Cited

# Next Steps





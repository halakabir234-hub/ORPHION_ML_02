Task-02: Image Generation using Stable Diffusion
Introduction

This project is completed as Task-02 of the ORPHION Internship Program.
The objective of this task is to generate artistic images from textual prompts using a diffusion-based generative AI model, specifically Stable Diffusion.

The implementation uses TensorFlow and KerasCV, following the official TensorFlow Generative AI guidelines, to demonstrate how modern text-to-image systems convert natural language descriptions into high-quality visual content.

Understanding Stable Diffusion

Stable Diffusion is a latent diffusion model that generates images by gradually denoising random noise while being guided by a text prompt. Instead of operating directly in pixel space, the model works in a compressed latent space, making it both memory-efficient and computationally fast.

The Stable Diffusion architecture consists of three main components:

A text encoder that converts the input prompt into a numerical representation

A diffusion model that iteratively denoises latent representations

A decoder that converts the final latent output into a high-resolution image

By conditioning the diffusion process on text embeddings, the model learns to generate images that closely align with the provided prompt.

Technology Stack

This project uses the following technologies:

Python

TensorFlow

KerasCV

Stable Diffusion

Google Colab (GPU runtime)

Environment Setup

To begin, the required dependencies are installed in Google Colab.

pip install tensorflow keras_cv --upgrade --quiet


The necessary libraries are then imported.

import time
import keras_cv
from tensorflow import keras
import matplotlib.pyplot as plt

Model Initialization

The Stable Diffusion model is initialized using KerasCV’s prebuilt implementation, configured to generate images at a resolution of 512×512 pixels.

model = keras_cv.models.StableDiffusion(
    img_width=512,
    img_height=512
)


By using this model checkpoint, usage is subject to the CreativeML Open RAIL-M License.

Text-to-Image Generation

Once the model is loaded, images can be generated directly from a text prompt. The model takes a prompt as input and produces visually rich images that align with the given description.

images = model.text_to_image(
    "photograph of an astronaut riding a horse",
    batch_size=3
)

Visualizing Generated Images

To display the generated images, a simple plotting function is used.

def plot_images(images):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")

plot_images(images)

Prompt Engineering and Creativity

Prompt engineering plays a crucial role in controlling the style, quality, and creativity of generated images. More descriptive prompts tend to produce more detailed and artistic results.

images = model.text_to_image(
    "cute magical flying dog, fantasy art, golden color, "
    "high quality, highly detailed, elegant, sharp focus, "
    "digital painting, mystery, adventure",
    batch_size=3
)

plot_images(images)

Performance Optimization

KerasCV’s Stable Diffusion implementation offers performance advantages such as mixed precision computation and XLA compilation, enabling faster image generation on supported GPUs.

Mixed Precision
keras.mixed_precision.set_global_policy("mixed_float16")

XLA Compilation
model = keras_cv.models.StableDiffusion(jit_compile=True)


When combined, these optimizations significantly reduce inference time while maintaining output quality.

Key Learnings

Through this task, the following concepts were explored:

Latent diffusion models and their working principles

Text-to-image generation using Stable Diffusion

Prompt engineering techniques

Performance optimization using TensorFlow features

Conclusion

This project demonstrates the practical implementation of Stable Diffusion using TensorFlow and KerasCV for text-to-image generation. By leveraging diffusion models and prompt engineering, it is possible to generate high-quality and creative images from simple textual descriptions. The task highlights the power of modern generative AI models and their real-world applications in creative and technical domains.

Reference

TensorFlow Generative AI – High-Performance Image Generation using Stable Diffusion in KerasCV
Authors: François Chollet, Luke Wood, Divam Gupta

Author

Hala Kabir
ORPHION Internship – Task-02

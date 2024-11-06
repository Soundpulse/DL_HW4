import torch
import matplotlib.pyplot as plt
from diffusers import DDIMPipeline
import random


pipeline = DDIMPipeline.from_pretrained("google/ddpm-celebahq-256")

generator = torch.Generator().manual_seed(88)

def sample_and_plot(steps):
    """
    Generates and plots an image using a diffusion model, saving the final image as a .png file.

    Args:
        steps (int): The number of inference steps to use in the diffusion process.

    Example:
        sample_and_plot(steps=50)
    """
    with torch.no_grad():
        generated_output = pipeline(generator=generator, num_inference_steps=steps, output_type="np")
        generated_image = generated_output.images[0]
    
    # Plot the generated image
    final_image = (generated_image + 1) / 2
    plt.imshow(final_image)
    plt.axis('off')
    plt.title(f'DDIM with {steps} steps')
    
    filename = f"ddim_{steps}_steps.png"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

# PARAMS USED: 10, 50, 100, 500
# RAN INDIVIDUALLY TO ENSURE LATENT CONSISTENCY
sample_and_plot(500)

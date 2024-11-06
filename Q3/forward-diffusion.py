import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from PIL import Image

# Settings
num_steps = 500
beta_start = 0.0001
beta_end = 0.02 

# Define the noise schedule using alpha
beta = torch.linspace(beta_start, beta_end, num_steps)
alpha = 1 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

# Convert images into tensors
img_dir = os.getcwd() + "/diffusion-input"
transform = transforms.ToTensor()

clean_images = []
for filename in os.listdir(img_dir)[:5]:
    img_path = os.path.join(img_dir, filename)
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img)
    clean_images.append(img_tensor)

clean_images = torch.stack(clean_images)

def forward_diffusion_process(x_0, num_steps, alpha_bar):
    """
    Forward diffusion process: progressively add Gaussian noise to images.
    
    Args:
        x_0 (torch.Tensor): Clean images tensor with shape (num_images, C, H, W).
        num_steps (int): Number of diffusion steps.
        alpha_bar (torch.Tensor): Cumulative product of alpha terms.
        
    Returns:
        list: Noisy images at each timestep.
    """
    x_t_samples = [x_0]
    
    for t in range(1, num_steps + 1):
        noise = torch.randn_like(x_0)
        sqrt_alpha_bar = torch.sqrt(alpha_bar[t-1])
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar[t-1])
        
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        x_t_samples.append(x_t)
    
    return x_t_samples

samples = forward_diffusion_process(clean_images, num_steps, alpha_bar)

mse_values = []
for t, x_t in enumerate(samples):
    mse = torch.mean((x_t - clean_images) ** 2).item()
    mse_values.append(mse)

# Plot the original and noisy images at specific timesteps
timesteps_to_show = [0, 10, 50, 100, 500]

fig, axes = plt.subplots(5, len(timesteps_to_show), figsize=(15, 10))
for i in range(5):
    for j, t in enumerate(timesteps_to_show):
        img = samples[t][i].permute(1, 2, 0).clip(0, 1)
        axes[i, j].imshow(img)
        axes[i, j].axis("off")
        if i == 0:
            axes[i, j].set_title(f"t = {t}\nMSE: {mse_values[t]:.4f}")

plt.suptitle("Images at Different Timesteps", fontsize=16)
plt.show()

# Plot MSE over all timesteps to show level of corruption
plt.figure(figsize=(8, 6))
plt.plot(range(num_steps + 1), mse_values, label="MSE between images")
plt.xlabel("Timestep")
plt.ylabel("Mean Squared Error")
plt.title("MSE at Different Timesteps")
plt.legend()
plt.grid(True)
plt.show()
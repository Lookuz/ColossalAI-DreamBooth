from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained("./stable-diffusion-v1-4", torch_dtype=torch.float16).to("cuda")
# Load finetuned personalization weights
state_dict = torch.load("./weights/diffusion_pytorch_model.bin")
pipe.unet.load_state_dict(state_dict)

prompt = "a gsipt character with a shield"
image = pipe(prompt, num_inference_steps=100, guidance_scale=7.5).images[0]

image.save("outputs/albedo.png")
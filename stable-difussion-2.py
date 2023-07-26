from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
import torch

model_id_or_path = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float32,)



init_image = Image.open("eren.jpg").convert("RGB").resize((768, 512))
prompt = "Transform into a super hero"

images = pipe(prompt=prompt, image=init_image).images
images[0].save("eren_super_hero.png")

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

# Carica la pipeline di Stable Diffusion XL (img2img)
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",  # Nome del modello
    torch_dtype=torch.float16,  # Usa float16 per GPU, float32 per CPU
).to("cuda")  # Usa "cpu" se non hai una GPU

# Carica un'immagine iniziale (input img2img)
init_image = Image.open("00045.png").convert("RGB")
init_image = init_image.resize((1024, 1024))  # Ridimensiona l'immagine a 1024x1024

# Definisci il prompt e i parametri
prompt = "A beautiful futuristic cityscape at sunset, highly detailed, ultra realistic"
strength = 0.6  # Determina quanto l'immagine iniziale influenza il risultato (0.0 = solo prompt, 1.0 = solo immagine)
guidance_scale = 7.5  # Controlla la "creatività" (valori tipici: 7-10)

# Genera l'immagine con img2img
output = pipe(prompt=prompt, image=init_image, strength=strength, guidance_scale=guidance_scale)

# Salva il risultato
output.images[0].save("immagine_output.png")

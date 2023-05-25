import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from tqdm.auto import tqdm
from PIL import Image

import pdb


def main():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, revision="fp16", torch_dtype=torch.float16
    )

    # extract all components from SD model
    vae = pipe.vae
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    unet = pipe.unet
    scheduler = pipe.scheduler

    # set all networks to gpu
    device = "cuda"
    vae.to(device)
    text_encoder.to(device)
    unet.to(device)

    # set params
    height = 512  # default height of Stable Diffusion
    width = 512  # default width of Stable Diffusion
    num_inference_steps = 100  # Number of denoising steps
    guidance_scale = 7.5  # Scale for classifier-free guidance
    generator = torch.manual_seed(0)  # Seed generator to create the inital latent noise

    prompt = ["a photo of an astronaut riding a horse on mars"]
    batch_size = len(prompt)

    # compute conditioning vector
    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # generate initial random noise for input [B, 4, 64, 64]
    latents = torch.randn(
        (batch_size, unet.config.in_channels, height // 8, width // 8),
        generator=generator,
        dtype=torch.float32,
        device=device,
    )

    scheduler.set_timesteps(num_inference_steps)
    latents = latents * scheduler.init_noise_sigma

    # denoising
    scheduler.set_timesteps(num_inference_steps)
    for t in tqdm(scheduler.timesteps):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(latents).sample

    # convert image to PIL
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    pil_images[0]

    # pdb.set_trace()

    pil_images.save("astronaut_rides_horse.png")


if __name__ == "__main__":
    main()
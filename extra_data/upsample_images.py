import os
from PIL import Image
from diffusers import StableDiffusionUpscalePipeline, LDMSuperResolutionPipeline
from super_image import EdsrModel, ImageLoader
import torch
import cv2


def upsample_folder_stabilityai(img_dir: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    # load model and scheduler
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    )
    pipeline = pipeline.to("cuda")

    for img_path in os.listdir(img_dir):
        if os.path.splitext(img_path)[-1] in [".png", ".jpeg", ".jpg"]:
            print(f"Processing {img_path}...")
            low_res_img = Image.open(os.path.join(img_dir, img_path)).convert("RGB")

            upscaled_image = pipeline(
                prompt="a mosquito on human skin", image=low_res_img
            ).images[0]

            save_path = os.path.join(save_dir, img_path)
            upscaled_image.save(save_path)

            print(f"Saved image to {save_path}...")


def upsample_folder_compvis(img_dir: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "CompVis/ldm-super-resolution-4x-openimages"
    # load model and scheduler
    pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id)
    pipeline = pipeline.to(device)

    for img_path in os.listdir(img_dir):
        if os.path.splitext(img_path)[-1] in [".png", ".jpeg", ".jpg"]:
            print(f"Processing {img_path}...")
            low_res_img = Image.open(os.path.join(img_dir, img_path)).convert("RGB")

            upscaled_image = pipeline(
                low_res_img, num_inference_steps=100, eta=1
            ).images[0]

            save_path = os.path.join(save_dir, img_path)
            upscaled_image.save(save_path)

            print(f"Saved image to {save_path}...")


def upsample_opencv(img_dir: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    for img_path in os.listdir(img_dir):
        if os.path.splitext(img_path)[-1] in [".png", ".jpeg", ".jpg"]:
            print(f"Processing {img_path}...")

            low_res_img = cv2.imread(
                os.path.join(img_dir, img_path), cv2.IMREAD_UNCHANGED
            )
            w, h = low_res_img.shape[:2]
            upscaled_image = cv2.resize(
                low_res_img, (4 * w, 4 * h), interpolation=cv2.INTER_LANCZOS4
            )

            save_path = os.path.join(save_dir, img_path)
            cv2.imwrite(save_path, upscaled_image)
            print(f"Saved image to {save_path}...")


def upsample_edsr(img_dir: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    for img_path in os.listdir(img_dir):
        if os.path.splitext(img_path)[-1] in [".png", ".jpeg", ".jpg"]:
            print(f"Processing {img_path}...")
            low_res_img = Image.open(os.path.join(img_dir, img_path)).convert("RGB")

            model = EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=4)
            inputs = ImageLoader.load_image(low_res_img)
            preds = model(inputs)

            save_path = os.path.join(save_dir, img_path)
            ImageLoader.save_image(preds, save_path)
            print(f"Saved image to {save_path}...")


img_dir = "../Mosquito-on-human-skin/Aedes aegypti smashed"
save_dir = "../Mosquito-on-human-skin/Aedes_aegypti_smashed_compvis_x4"

upsample_folder_stabilityai(img_dir, save_dir)

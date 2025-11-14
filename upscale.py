import os
import requests
from PIL import Image
from realesrgan import RealESRGAN
import torch

def download_model():
    model_file = "RealESRGAN_x4plus.pth"
    model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5/RealESRGAN_x4plus.pth"

    if not os.path.exists(model_file):
        print("Model not found... downloading now...")
        try:
            r = requests.get(model_url, allow_redirects=True, stream=True)
            with open(model_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print("Model downloaded successfully.")
        except Exception as e:
            print("Error while downloading model:", e)
    else:
        print("Model already exists. Using the existing file.")


def upscale_folder(input_folder, output_folder):
    try:
        # download model if not present
        download_model()

        # gpu or cpu check
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load model
        model = RealESRGAN(device, scale=4)
        model.load_weights("RealESRGAN_x4plus.pth")

        # create output folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # loop images
        for file_name in os.listdir(input_folder):
            if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                input_path = os.path.join(input_folder, file_name)
                output_path = os.path.join(output_folder, "4k_" + file_name)

                print("Upscaling:", file_name)
                img = Image.open(input_path)

                # upscale
                result = model.predict(img)

                # save output
                result.save(output_path)
                print("Saved:", output_path)

        print("All images processed successfully.")

    except Exception as e:
        print("Error:", e)


# give your folders here
input_dir = "my_images"
output_dir = "upscaled_images"

upscale_folder(input_dir, output_dir)

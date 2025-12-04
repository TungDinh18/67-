import requests
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io

# ==============================
# CONFIG
# ==============================
HF_API_KEY = "YOUR_HUGGINGFACE_API_KEY"
MODEL_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"

headers = {
    "Authorization": f"Bearer {HF_API_KEY}"
}

# ==============================
# IMAGE GENERATION
# ==============================
def generate_image(prompt):
    payload = {
        "inputs": prompt
    }

    response = requests.post(MODEL_URL, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception("Image generation failed:", response.text)

    image_bytes = response.content
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image.save("original.png")
    print("Saved: original.png")
    return image


# ==============================
# DAYLIGHT EDITION
# ==============================
def daylight_edition(image):
    enhancer_brightness = ImageEnhance.Brightness(image)
    bright_img = enhancer_brightness.enhance(1.3)

    enhancer_contrast = ImageEnhance.Contrast(bright_img)
    soft_img = enhancer_contrast.enhance(0.9)

    soft_img.save("daylight_edition.png")
    print("Saved: daylight_edition.png")


# ==============================
# NIGHT MOOD EDITION
# ==============================
def night_mood(image):
    cv_img = np.array(image)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

    # Increase contrast
    alpha = 1.4  # contrast
    beta = -30   # brightness
    contrast_img = cv2.convertScaleAbs(cv_img, alpha=alpha, beta=beta)

    # Subtle Gaussian blur
    blurred = cv2.GaussianBlur(contrast_img, (7, 7), 0)

    final = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
    Image.fromarray(final).save("night_mood.png")
    print("Saved: night_mood.png")


# ==============================
# MAIN PROGRAM
# ==============================
if __name__ == "__main__":
    prompt = input("Enter your AI image prompt: ")

    base_image = generate_image(prompt)

    daylight_edition(base_image)
    night_mood(base_image)

    print("\nAll versions generated successfully!")

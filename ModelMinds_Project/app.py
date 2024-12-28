from flask import Flask, render_template, request, send_file
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch
from io import BytesIO
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Load GPT model for text generation
generator = pipeline("text-generation", model="gpt-2")

# Load Stable Diffusion pipeline for image generation
stable_diffusion = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v-1-4-original")
stable_diffusion.to("cuda")

@app.route("/", methods=["GET", "POST"])
def home():
    character_description = None
    character_image = None

    if request.method == "POST":
        # Get character traits from user input
        character_traits = request.form.get("traits")

        # Generate a character description using GPT
        description = generator(character_traits, max_length=50)[0]["generated_text"]
        character_description = description

        # Generate a character image using Stable Diffusion based on description
        image = stable_diffusion(description).images[0]

        # Convert the image to a byte stream to serve it via Flask
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)
        character_image = img_byte_arr.getvalue()  # Convert byte stream to raw byte data

        return render_template("index.html", description=character_description, image=character_image)

    return render_template("index.html", description=character_description, image=character_image)

@app.route("/image")
def image():
    return send_file(character_image, mimetype="image/png")

if __name__ == "__main__":
    app.run(debug=True)

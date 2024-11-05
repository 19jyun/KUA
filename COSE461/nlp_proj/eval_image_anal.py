from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from PIL import Image
import os

# Load image captioning model and tokenizer
image_captioning_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
caption_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

def generate_caption(image_path):
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values

    # Generate caption
    output_ids = image_captioning_model.generate(pixel_values, max_length=16, num_beams=4, early_stopping=True)
    caption = caption_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

def evaluate_images_in_directory(directory):
    for filename in os.listdir(directory):
        image_path = os.path.join(directory, filename)
        caption = generate_caption(image_path)
        print(f"Image: {image_path}")
        print(f"Caption: {caption}\n")

if __name__ == "__main__":
    evaluate_images_in_directory('image_context')
import torch
from PIL import Image
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
import matplotlib.pyplot as plt
import numpy as np
import os

print("=" * 60)
print("BLIP VISION-LANGUAGE APPLICATION")
print("=" * 60)
print()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print()

try:
    print("Loading BLIP models...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

def load_image(image_path):
    if image_path.startswith('http'):
        response = requests.get(image_path, stream=True)
        image = Image.open(response.raw)
    else:
        image = Image.open(image_path)
    return image.convert('RGB')

def generate_caption(image):
    inputs = processor(image, return_tensors="pt").to(device)
    out = caption_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def answer_question(image, question):
    inputs = processor(image, question, return_tensors="pt").to(device)
    out = vqa_model.generate(**inputs)
    answer = processor.decode(out[0], skip_special_tokens=True)
    return answer

def display_image_and_caption(image, caption):
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.title(f"Generated Caption: {caption}", fontsize=12, pad=10)
    plt.axis('off')
    plt.show()

def upload_image():
    print("\n" + "=" * 60)
    print("IMAGE SOURCE")
    print("=" * 60)
    print("1. Enter image path (local)")
    print("2. Enter image URL")
    print("3. Use default sample image")

    choice = input("\nSelect option (1/2/3): ")

    if choice == '1':
        path = input("Enter image path: ")
        if os.path.exists(path):
            return load_image(path)
        else:
            print("File not found. Using sample image.")
            return load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/sample_image.jpg")

    elif choice == '2':
        url = input("Enter image URL: ")
        try:
            return load_image(url)
        except:
            print("Invalid URL. Using sample image.")
            return load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/sample_image.jpg")

    else:
        print("Using sample image...")
        return load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/sample_image.jpg")

def ask_questions(image):
    print("\n" + "=" * 60)
    print("VISUAL QUESTION ANSWERING")
    print("=" * 60)

    sample_questions = [
        "What is in this image?",
        "What color is the main object?",
        "Is there a person in this image?",
        "What is the background?",
        "What is happening in this image?"
    ]

    print("\nSample questions:")
    for i, q in enumerate(sample_questions, 1):
        print(f"  {i}. {q}")

    print("\n6. Ask your own question")
    choice = input("\nSelect question (1-6): ")

    if choice == '6':
        question = input("Enter your question: ")
    else:
        try:
            question = sample_questions[int(choice) - 1]
        except:
            question = "What is in this image?"

    answer = answer_question(image, question)
    print(f"\nQ: {question}")
    print(f"A: {answer}")

def main_menu():
    print("\n" + "=" * 60)
    print("BLIP VISION-LANGUAGE MENU")
    print("=" * 60)
    print("1. Upload image and generate caption")
    print("2. Ask questions about an image")
    print("3. Both (Caption + Questions)")
    print("4. Exit")

    choice = input("\nSelect option (1-4): ")
    return choice

def main():
    print("\n" + "=" * 60)
    print("BLIP VISION-LANGUAGE APPLICATION")
    print("=" * 60)

    image = None

    while True:
        choice = main_menu()

        if choice == '4':
            print("\nGoodbye!")
            break

        if choice in ['1', '3']:
            image = upload_image()

            if image is None:
                print("Failed to load image. Try again.")
                continue

            caption = generate_caption(image)
            display_image_and_caption(image, caption)

            print(f"\nCaption: {caption}")

        if choice in ['2', '3']:
            if image is None:
                image = upload_image()

            if image is None:
                print("Failed to load image. Try again.")
                continue

            ask_questions(image)

        if choice not in ['1', '2', '3', '4']:
            print("Invalid choice. Please try again.")

    print("\n" + "=" * 60)
    print("SYSTEM READY")
    print("=" * 60)

if __name__ == "__main__":
    main()
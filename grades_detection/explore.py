
# %% [markdown]
# ## 1. load_libraries

#%%
import os
from pathlib import Path
from IPython.display import display
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import matplotlib.pyplot as plt
import numpy as np


print("PyTorch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())

#%%
IMAGE_FOLDER = "./data/raw/grades_images"
if not Path(IMAGE_FOLDER).exists():
    raise FileNotFoundError(f"Folder not found: {IMAGE_FOLDER}") 
print(f"✓ Image folder is find: {IMAGE_FOLDER}")

#%%
image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith('.png')]
print(f"Found {len(image_files)} PNG images:")


#%%

for i,filename in enumerate(image_files,1):
    print(f"{i} {filename}")

#%%
def read_image(file_name):
    image_path = os.path.join(IMAGE_FOLDER,file_name)
    img = Image.open(image_path)

    return {
        "filename": file_name,
        "size" : img.size,
        "width" : img.size[0],
        "height": img.size[1],
        "mode": img.mode,
        "format": img.format
    }
    
#%%
images_info = []
for filename in image_files:
    info = read_image(filename)
    images_info.append(info)
    
    print(f"\n{info['filename']}:")
    print(f"  Dimensions: {info['width']} x {info['height']} pixels")
    print(f"  Color mode: {info['mode']}")
    print(f"  Format: {info['format']}")

# %% 
if image_files:  # Check if list is not empty
    image_path = os.path.join(IMAGE_FOLDER, image_files[33])  
    img = Image.open(image_path)
    display(img)

# %% [markdown]
## Load Pretrained TrOCR Models
#%%
print("\n📥 Loading TrOCR model...")

# For HANDWRITTEN text (grades and names)
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

print("✅ Models loaded successfully!")

#%%
# Region Extraction Functions
def extract_name_region(image):
    """
    Extract the top-left region where the name is located
    """
    width, height = image.size
    # Top-left region: approximately first 15% of height, left 50% of width
    name_region = image.crop((0, 0, int(width * 0.5), int(height * 0.1)))
    return name_region

def extract_id_region(image):
    """
    Extract the top-right region where the ID is located
    """
    width, height = image.size
    # Top-right region: approximately first 15% of height, right 50% of width
    id_region = image.crop((int(width * 0.7), 0, width, int(height * 0.15)))
    return id_region
def extract_grade_region(image):
    """
    Extract the bottom-right region where the grade is located
    """
    width, height = image.size
    # Bottom-right corner: last 15% of height, right 20% of width
    grade_region = image.crop((int(width * 0.8), int(height * 0.85), width, height))
    return grade_region

#%%
# Visualize name regions only for sample images
for idx in [0, 33, 100]:
    image_path = os.path.join(IMAGE_FOLDER, image_files[idx])
    img = Image.open(image_path)
    
    # Extract name region
    name_region = extract_name_region(img)
    # id_region = extract_id_region(img)
    # grade_region = extract_grade_region(img)
    
    # Display
    plt.figure(figsize=(8, 2))
    plt.imshow(name_region)
    # plt.imshow(id_region)
    # plt.imshow(grade_region)
    plt.axis('off')
    plt.show()
#%%
# OCR Extraction Function
def extract_text_with_trocr(image_region):
    """
    Use TrOCR to extract text from an image region
    """
    # Preprocess image
    pixel_values = processor(image_region, return_tensors="pt").pixel_values
    
    # Generate text
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    
    # Decode to text
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return generated_text
#%%
def process_single_image(image_path, show_regions=False):
    """
    Process a single image to extract name, ID, and grade
    """
    # Load image
    img = Image.open(image_path)
    
    # Extract regions
    name_region = extract_name_region(img)
    id_region = extract_id_region(img)
    grade_region = extract_grade_region(img)
    
    # Extract text using TrOCR
    name_text = extract_text_with_trocr(name_region)
    id_text = extract_text_with_trocr(id_region)
    grade_text = extract_text_with_trocr(grade_region)
    
    # Display results
    if show_regions:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        axes[0, 0].imshow(img)
        axes[0, 0].set_title("Full Image")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(name_region)
        axes[0, 1].set_title(f"Name Region\nExtracted: {name_text}")
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(id_region)
        axes[1, 0].set_title(f"ID Region\nExtracted: {id_text}")
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(grade_region)
        axes[1, 1].set_title(f"Grade Region\nExtracted: {grade_text}")
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return name_text, id_text, grade_text

#%%
# test
print("\n" + "="*70)
print("TESTING PRETRAINED TrOCR ON YOUR IMAGES")
print("="*70)

# Test on images: 0, 33, 50, 100, 150, 200
test_indices = [0, 33, 50, 100, 150, 200]

results = []

for idx in test_indices:
    if idx < len(image_files):
        print(f"\n📄 Processing Image #{idx}: {image_files[idx]}")
        image_path = os.path.join(IMAGE_FOLDER, image_files[idx])
        
        name, student_id, grade = process_single_image(image_path, show_regions=True)
        
        results.append({
            'index': idx,
            'filename': image_files[idx],
            'name': name,
            'id': student_id,
            'grade': grade
        })
        
        print(f"   ✓ Name: {name}")
        print(f"   ✓ ID: {student_id}")
        print(f"   ✓ Grade: {grade}")


#%%
# Debug: Let's look at one specific image closely
idx = 33  # or any index you want to check
image_path = os.path.join(IMAGE_FOLDER, image_files[idx])
img = Image.open(image_path)

# Extract grade region
grade_region = extract_grade_region(img)

# Display it larger
plt.figure(figsize=(8, 6))
plt.imshow(grade_region)
plt.title("Grade Region - What does the model see?")
plt.axis('off')
plt.show()

# Extract text
grade_text = extract_text_with_trocr(grade_region)
print(f"Extracted grade text: '{grade_text}'")
#%%

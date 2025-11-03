
# %% [markdown]
# ## 1. load_libraries

#%%
import torch
from PIL import Image
import os
from pathlib import Path
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

# %% [markdown]
# ## 5. Summary Statistics

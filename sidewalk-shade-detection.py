from transformers import pipeline
from PIL import Image
import numpy as np
import requests
from io import BytesIO

# Specify image URL here
url = "https://private-user-images.githubusercontent.com/185747678/434916093-8741351c-29de-4b68-b2fb-cb5f278d7fe3.jpg?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDQ5MjUyODAsIm5iZiI6MTc0NDkyNDk4MCwicGF0aCI6Ii8xODU3NDc2NzgvNDM0OTE2MDkzLTg3NDEzNTFjLTI5ZGUtNGI2OC1iMmZiLWNiNWYyNzhkN2ZlMy5qcGc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwNDE3JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDQxN1QyMTIzMDBaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT03MjdmZTk1OTkxMzU3ZDZjMWY5NDBmNWY5OWNiZWY1YjBlYzA0ZWY2NTBmZmM1Y2JiOWE1Y2QzMjFlYjAwMTQzJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.ZGOCKBPny7lAyA1NZuDsh0oVdUZhTE9Ex92Ny-YZbuk"

# Load image directly from URL
response = requests.get(url)
image = Image.open(BytesIO(response.content)).convert("RGB")

# Hugging Face model
panoptic_model = "facebook/mask2former-swin-large-cityscapes-panoptic"

segmenter = pipeline("image-segmentation", model=panoptic_model)
results = segmenter(image)

sidewalk_masks = []
person_count = 0

# Extract sidewalk masks and count people
for result in results:
    label = result['label'].lower()
    if label == "sidewalk":
        sidewalk_masks.append(result['mask'])
    elif label == "person":
        person_count += 1

# Combine masks if there are multiple sidewalk segments
if sidewalk_masks:
    combined_mask = np.any([np.array(mask) for mask in sidewalk_masks], axis=0)

    # Convert original image to numpy array
    image_array = np.array(image)

    # Calculate average pixel value within sidewalk mask
    sidewalk_pixels = image_array[combined_mask]
    avg_pixel_value = sidewalk_pixels.mean(axis=0)

    print("Average sidewalk pixel value (RGB):", avg_pixel_value)

    # Calculate perceived brightness
    brightness = 0.2126 * avg_pixel_value[0] + 0.7152 * avg_pixel_value[1] + 0.0722 * avg_pixel_value[2]
    print("Average sidewalk brightness:", brightness)

    # Classify as shaded or unshaded based on threshold
    if brightness < 130:
        print("This sidewalk is likely SHADED.")
    else:
        print("This sidewalk is likely UNSHADED.")

    # Save combined mask as an image
    Image.fromarray((combined_mask * 255).astype(np.uint8)).save("sidewalk_mask.png")
else:
    print("No sidewalk segments detected.")

# Print the number of people detected
print("Number of people detected:", person_count)

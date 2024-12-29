import os
from PIL import Image

def resize_image(image_path, size=(512, 512)):
    # Get the image file name and directory
    directory, file_name = os.path.split(image_path)
    
    # Create 'resized' directory if it doesn't exist
    resized_dir = os.path.join(directory, 'resized_512x512')
    if not os.path.exists(resized_dir):
        os.makedirs(resized_dir)
    
    # Open the image
    with Image.open(image_path) as img:
        # Resize the image
        resized_img = img.resize(size)
        
        # Construct the output path
        output_path = os.path.join(resized_dir, file_name)
        
        # Save the resized image
        resized_img.save(output_path)
        print(f"Image saved at {output_path}")

# Example usage
image_path = "app/loreal_images/bottles_comb_resized.jpg"  # Replace this with the path to your image
resize_image(image_path)

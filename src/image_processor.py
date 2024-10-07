import os
from PIL import Image

def load_images(image_folder):
    images = []
    for filename in os.listdir(image_folder):
        if filename.endswith(('png', 'jpg', 'jpeg', 'webp')):
            img_path = os.path.join(image_folder, filename)
            img = Image.open(img_path)
            images.append((img, filename))
            print(f"Loaded image: {filename}")
    return images

# Test loading images from a folder
image_folder = './images'  # Path to your folder with images
images = load_images(image_folder)

def resize_image(image, target_size):
    resized_image = image.resize(target_size, Image.ANTIALIAS)  # ANTIALIAS for high-quality downscaling
    return resized_image

# Test resizing an image
target_size = (300, 300)  # Example target size (width, height)
resized_image = resize_image(images[0][0], target_size)
resized_image.show()  # Display the resized image (for testing)

def save_image_in_formats(image, output_folder, filename, format='webp', quality=85):
    output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.{format}")
    
    if format.lower() == 'webp':
        image.save(output_path, 'webp', quality=quality)
    elif format.lower() == 'png':
        image.save(output_path, 'png')
    elif format.lower() == 'jpeg':
        image.save(output_path, 'jpeg', quality=quality)
    else:
        raise ValueError(f"Unsupported format: {format}")

    print(f"Saved image: {output_path}")

# Test saving an image in WebP format
output_folder = './output'  # Path to your output folder
save_image_in_formats(resized_image, output_folder, images[0][1], format='webp', quality=85)
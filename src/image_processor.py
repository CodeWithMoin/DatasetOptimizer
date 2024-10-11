import os
import cv2
import numpy as np
from PIL import Image
import concurrent.futures


def load_images(image_folder):
    """Load images from a specified folder."""
    images = []
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('png', 'jpg', 'jpeg', 'webp')):
            img_path = os.path.join(image_folder, filename)
            img = Image.open(img_path)
            images.append((img, filename))
            print(f"Loaded image: {filename}")
    return images

def resize_image(image, target_size):
    """Resize an image to the target size."""
    resized_image = image.resize(target_size, Image.LANCZOS)  # LANCZOS for high-quality downscaling
    return resized_image

def save_image_in_formats(image, output_folder, filename, format='webp', quality=85):
    """Save the image in the specified format."""
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

def process_images(image_dir, output_dir, target_size, output_format, rotation_angle=None, scale_factor=None):
    images = load_images(image_dir)
    
    for img, img_path in images:  # Correct unpacking
        img = cv2.imread(os.path.join(image_dir, img_path))  # Read image correctly

        # Apply affine transformations if specified
        if rotation_angle is not None:
            img = rotate_image(img, rotation_angle)
        if scale_factor is not None:
            img = scale_image(img, scale_factor)

        # Resize the image to the target size
        img_resized = cv2.resize(img, target_size)

        # Save the processed image
        base_name = os.path.basename(img_path)
        output_path = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}.{output_format}")
        cv2.imwrite(output_path, img_resized)
        print(f"Saved image: {output_path}")


def rotate_image(image, angle):
    """Rotate the image by a specified angle."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Create the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def scale_image(image, scale_factor):
    """Scale the image by a specified scale factor."""
    new_size = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
    scaled = cv2.resize(image, new_size)
    return scaled

def batch_process_images(image_dir, output_dir, batch_size=10, target_size=(256, 256), output_format='png'):
    images = load_images(image_dir)
    num_batches = len(images) // batch_size + (len(images) % batch_size > 0)

    for i in range(num_batches):
        batch_images = images[i * batch_size:(i + 1) * batch_size]
        
        for img, img_path in batch_images:  # Correct unpacking
            img = cv2.imread(os.path.join(image_dir, img_path))  # Read image correctly
            img_resized = cv2.resize(img, target_size)
            base_name = os.path.basename(img_path)
            output_path = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}.{output_format}")
            cv2.imwrite(output_path, img_resized)

        print(f'Processed batch {i + 1}/{num_batches}')

def process_single_image(img_path, output_dir, target_size, output_format, rotate=None, scale=None):
    # Read the image using OpenCV (this returns a NumPy array)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Error: Unable to load image {img_path}")
        return

    # Apply rotation if specified
    if rotate is not None:
        img = rotate_image(img, rotate)

    # Apply scaling if specified
    if scale is not None:
        img = scale_image(img, scale)

    # Resize the image without borders
    img_resized = resize_image_opencv(img, target_size)

    # Construct output path and save the image in the desired format
    base_name = os.path.basename(img_path)
    output_path = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}.{output_format}")
    cv2.imwrite(output_path, img_resized)
    print(f"Saved image: {output_path}")

def parallel_process_images(image_dir, output_dir, target_size=(256, 256), output_format='png', rotate=None, scale=None, workers=4):
    images = load_images(image_dir)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for img, img_filename in images:  # Correct unpacking here
            img_path = os.path.join(image_dir, img_filename)
            futures.append(executor.submit(process_single_image, img_path, output_dir, target_size, output_format, rotate, scale))
        
        # Wait for all futures to complete
        for future in concurrent.futures.as_completed(futures):
            future.result()  # Get the result to raise exceptions if any

def resize_image_opencv(image, target_size):
    """Resize an OpenCV image while maintaining the aspect ratio."""
    original_height, original_width = image.shape[:2]
    target_width, target_height = target_size

    # Calculate the aspect ratio for scaling
    aspect_ratio = min(target_width / original_width, target_height / original_height)

    # Calculate new dimensions based on the aspect ratio
    new_width = int(original_width * aspect_ratio)
    new_height = int(original_height * aspect_ratio)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

    # If you want exact target size, create a canvas and paste the resized image in the center
    if (new_width, new_height) != (target_width, target_height):
        canvas = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255  # White canvas
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image
        resized_image = canvas

    return resized_image

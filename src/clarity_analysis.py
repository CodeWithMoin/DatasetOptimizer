import os
import cv2
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import numpy as np

# Paths
original_images_dir = '/Users/moinuddinshaik/Downloads/VS CODE/ML/DatasetOptimizer/data/subset'
resized_images_dir = '/Users/moinuddinshaik/Downloads/VS CODE/ML/DatasetOptimizer/data/resized'
output_csv_path = './data/clarity_metrics.csv'

def calculate_psnr(original, resized):
    """Calculate PSNR between two images."""
    mse = np.mean((original - resized) ** 2)
    if mse == 0:
        return float('inf')  # No noise, return infinite PSNR
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def calculate_ssim(original, resized):
    """Calculate SSIM between two images."""
    return ssim(original, resized, multichannel=True, win_size=3)

def calculate_mse(original, resized):
    """Calculate MSE between two images."""
    return np.mean((original - resized) ** 2)

def calculate_nmse(original, resized):
    """Calculate NMSE between two images."""
    mse = calculate_mse(original, resized)
    norm = np.mean(original ** 2)
    return mse / norm

def calculate_snr(original, resized):
    """Calculate SNR between two images."""
    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - resized) ** 2)
    return 10 * np.log10(signal_power / noise_power)

def calculate_cr(original, resized):
    """Calculate Compression Ratio (CR) based on file sizes."""
    original_size = os.path.getsize(original)
    resized_size = os.path.getsize(resized)
    return original_size / resized_size

# List to store metrics
metrics = []

# Iterate through the original images
for img_file in os.listdir(original_images_dir):
    if img_file.lower().endswith(('png', 'jpg', 'jpeg', 'webp')):
        original_path = os.path.join(original_images_dir, img_file)

        # Construct the path of the resized image directly
        resized_path = os.path.join(resized_images_dir, img_file)  # Assuming the same filename and format

        if os.path.exists(resized_path):
            resized = cv2.imread(resized_path)

            # Load the original image
            original = cv2.imread(original_path)

            if original is not None and resized is not None:
                # **Upscale** the resized image back to the original resolution
                upscaled_image = cv2.resize(resized, (original.shape[1], original.shape[0]))

                psnr_value = calculate_psnr(original, upscaled_image)
                ssim_value = calculate_ssim(original, upscaled_image)
                mse_value = calculate_mse(original, upscaled_image)
                nmse_value = calculate_nmse(original, upscaled_image)
                snr_value = calculate_snr(original, upscaled_image)
                cr_value = calculate_cr(original_path, resized_path)

                # Append all results to the list
                metrics.append({
                    'image': img_file,
                    'PSNR': psnr_value,
                    'SSIM': ssim_value,
                    'MSE': mse_value,
                    'NMSE': nmse_value,
                    'SNR': snr_value,
                    'CR': cr_value
                })
                print(f"Processed {img_file}: PSNR = {psnr_value:.2f}, SSIM = {ssim_value:.4f}, MSE = {mse_value:.4f}, NMSE = {nmse_value:.4f}, SNR = {snr_value:.2f}, CR = {cr_value:.2f}")
            else:
                print(f"Error loading images for {img_file}")
        else:
            print(f"No resized image found for: {img_file}")

# Convert metrics list to a DataFrame
metrics_df = pd.DataFrame(metrics)

# Save to CSV
metrics_df.to_csv(output_csv_path, index=False)
print(f"Clarity metrics saved to {output_csv_path}")

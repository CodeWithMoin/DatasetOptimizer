import os
import argparse
from image_processor import load_images, process_single_image, rotate_image, scale_image, parallel_process_images

# Main function to parse arguments and run the processing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images in a dataset.")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input images.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save processed images.')
    parser.add_argument('--resize', type=str, required=True, help='Target resolution (e.g., 256x256).')
    parser.add_argument('--format', type=str, choices=['png', 'jpeg', 'webp'], default='png', help='Output format.')
    parser.add_argument('--rotate', type=float, default=0.0, help='Rotation angle in degrees.')
    parser.add_argument('--scale', type=float, default=1.0, help='Scale factor (e.g., 1.0 for no scaling).')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers.')

    args = parser.parse_args()

    # Parse the resize argument
    width, height = map(int, args.resize.split('x'))
    target_size = (width, height)

    # Run the image processing
    parallel_process_images(args.input_dir, args.output_dir, target_size, args.format, args.rotate, args.scale, args.workers)

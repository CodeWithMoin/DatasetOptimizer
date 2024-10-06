# DatasetOptimizer

**DatasetOptimizer** is a Python-based tool designed to efficiently reduce the size of image datasets while maintaining optimal clarity and quality. This project is particularly aimed at machine learning practitioners who need to preprocess images for training models, ensuring that datasets remain manageable in size without sacrificing detail.

## Features

- **Batch Processing**: Load and process multiple images at once, streamlining the workflow for large datasets.
- **Image Resizing**: Resize images while maintaining aspect ratios to ensure visual consistency.
- **Multiple Output Formats**: Save processed images in various formats, including PNG, JPEG, and WebP, with configurable quality settings.
- **Affine Transformations**: Apply transformations (like rotation and scaling) to images, enhancing the diversity of your dataset.
- **Quality Retention**: Utilize advanced algorithms to ensure that the reduced images retain high quality and clarity, making them suitable for machine learning applications.

## Installation

To get started with DatasetOptimizer, clone this repository and install the required packages:

```bash
git clone https://github.com/CodeWithMoin/DatasetOptimizer.git
cd DatasetOptimizer
pip install -r requirements.txt

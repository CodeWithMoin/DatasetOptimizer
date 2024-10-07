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
```

### Step 2: Install dependencies

Install the required Python libraries using pip:
```bash
pip install -r requirements.txt
```

This will install:

- **Pillow** (for image processing)
- **OpenCV** (for resizing and transformations)
- **Numpy** (for matrix operations)
- **scikit-image** (for additional image processing utilities)
- **pycocotools** (if you’re using the COCO dataset)

## Usage

### **Step 1**: Place your dataset in the data/ folder

Put the images you want to compress in the data/ directory, either as a single folder or organized in subfolders.

### **Step 2**: Run the script

Modify the parameters in the src/main.py file to set your desired output size, formats, and transformation settings.

For example, you can configure the following:


- **Output resolution**: Target size for the resized images.
- **Output format**: Choose between PNG, JPEG, or WebP.
- **Quality**: Set compression quality for formats like JPEG.

Once configured, run the main script:
```bash
python src/main.py
```
The processed images will be saved in the output/ folder, maintaining their original structure (subfolders, etc.).

### Example
```bash
python src/main.py --input_dir data/ --output_dir output/ --resize 256x256 --format png
```
This will resize all images in the data/ directory to 256x256 and save them in PNG format in the output/ directory

## Configuration Options

You can configure various options in main.py or via command-line arguments:

- **--input_dir**: Directory containing input images.
- **--output_dir**: Directory to save the processed images.
- **--resize**: Desired resolution (e.g., 256x256, 512x512).
- **--format**: Output format (png, jpeg, webp).
- **--quality**: Compression quality for formats like JPEG/WebP (range 1-100).
- **--transform**: Apply affine transformations (rotate, scale, etc.).

## Datasets

While this tool can process any image dataset, it is designed with machine learning datasets in mind. You can easily use it with popular datasets like:

- **COCO Dataset** (Common Objects in Context)
- **ImageNet**
- **Open Images**
- **CIFAR-10**
- **Fashion-MNIST**

## Contributing

Contributions are welcome! If you have ideas to improve DatasetOptimizer, feel free to submit a pull request or open an issue. Whether it’s a bug fix, feature suggestion, or performance improvement, we appreciate any input.

To contribute:

1. Fork the repository.
2. Create a feature branch (git checkout -b feature-branch).
3. Commit your changes (git commit -m 'Add some feature').
4. Push to the branch (git push origin feature-branch).
5. Open a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- This project was inspired by the need to optimize large datasets for machine learning workflows, minimizing resource use while preserving image clarity.
- Thanks to the developers of Pillow, OpenCV, and other libraries that make image processing in Python easy and powerful.

**Happy optimizing your datasets!**

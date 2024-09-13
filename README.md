### Hyperbolic Autoencoder for Image Compression

This repository contains code for compressing and reconstructing images using a convolutional autoencoder enhanced with a hyperbolic projection layer. The model incorporates several advanced techniques to reduce loss in compression without sacrificing performance, making it suitable for compressing one image at a time.

### Table of Contents
- Features
- Requirements
- Installation
- Usage
- Code Overview
- Important Notes
- Troubleshooting
- Contributing
- License
- Acknowledgments

### Features

- Hyperbolic Projection Layer: Projects data into hyperbolic space to capture hierarchical relationships and improve compression.
- Convolutional Autoencoder: Utilizes convolutional layers to efficiently encode and decode image data.
- Residual and Skip Connections: Enhances information flow and improves reconstruction quality.
- Advanced Loss Function: Implements Structural Similarity Index Measure (SSIM) loss to focus on perceptual similarity.
- Regularization Techniques: Includes Batch Normalization and Dropout to prevent overfitting.
- Data Augmentation: Generates a diverse dataset from a single image using augmentation techniques.
- Learning Rate Scheduling: Adjusts the learning rate during training to improve convergence.
- Early Stopping and Model Checkpointing: Prevents overfitting and saves the best-performing model.

### Requirements:

- Python: 3.7 or higher
- TensorFlow: 2.x
- NumPy
- Matplotlib
- Pillow
- TensorFlow Model Optimization Toolkit: (Optional, if attempting quantization)

### Installation

Clone the Repository:

git clone https://github.com/wanderingacorn112/imgCompressor.git
cd imgCompressor

Create a virtual environment (optional):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install the required packages:

pip install tensorflow numpy matplotlib pillow

For quantization (optional):

pip install tensorflow-model-optimization

### Usage:

Prepare your image

Place the image you want to compress in a data/ directory within the repository.
For example: data/Mahayana.png

Open main.py and update the image_path variable:

image_path = 'data/Mahayana.png'  # Replace with your image path

Run the script

python main.py

View the results

The script will display the original and reconstructed images side by side.
The compressed latent space representation will be printed in the console.

### Code Overview

HyperbolicProjection Layer

Custom Keras layer that projects input data into hyperbolic space.
Includes a get_config method for serialization.
Data Augmentation

Uses ImageDataGenerator to create a dataset from a single image.
Applies transformations like rotation, flipping, zooming, and shifting.
Model Architecture

Encoder

Convolutional layers with increasing filters (64, 128, 256).
Includes Batch Normalization and Leaky ReLU activation.
Skip connections are saved for the decoder.
Latent Space

Dense layers reduce dimensionality to create a compressed representation.
Decoder

Mirrors the encoder with UpSampling and convolutional layers.
Skip connections from the encoder are added to corresponding layers.
Reconstructs the image from the latent space.
Training

Compiled with the Adam optimizer and SSIM loss function.
Includes callbacks:
EarlyStopping: Stops training when validation loss stops improving.
ModelCheckpoint: Saves the best model during training.
ReduceLROnPlateau: Reduces learning rate when a plateau in validation loss is detected.
TensorBoard: Logs metrics for visualization.
Evaluation

After training, the model compresses and reconstructs the original image.
Displays the original and reconstructed images.
Outputs the latent space representation.

### Important Notes:

Single Image Training

The model is designed to compress one image at a time.
Data augmentation is used to create a dataset from the single image, which helps the model learn effectively.
Quantization

Quantization is excluded due to compatibility issues with custom layers.
The focus is on other optimization techniques to reduce loss without sacrificing performance.
Model Saving

The model is saved using the .keras format, which is the new standard in TensorFlow and Keras.
Ensure that the filepath ends with .keras when saving and loading the model.
TensorFlow and Keras Versions

The code is compatible with TensorFlow 2.x.
Ensure that all Keras layers are imported from tensorflow.keras.layers.


### Troubleshooting:

AttributeError: 'NumpyArrayIterator' object has no attribute 'next'

Solution: Replace data_generator.next() with next(data_generator) in the create_dataset_from_image function.
ValueError: The filepath provided must end in .keras

Solution: Update the ModelCheckpoint filepath to end with .keras and use .keras when loading the model.

model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True)


Issues with Custom Layers and Quantization

Quantization can cause compatibility issues with custom layers like HyperbolicProjection.
Exclude custom layers from quantization or focus on other optimization techniques.


### Contributions:

Contributions are welcome! If you have ideas for improvements or encounter any issues, feel free to open an issue or submit a pull request.

Fork the repository.

Create a new branch:

git checkout -b feature/YourFeature

git commit -m "Add YourFeature"

git push origin feature/YourFeature

Open a pull request

### License

This project is licensed under the MIT License. See the LICENSE file for details.

### Acknowledgments

TensorFlow and Keras Teams: For providing powerful tools and libraries for machine learning.
Community Contributors: For ideas and inspiration on optimizing neural networks for compression.

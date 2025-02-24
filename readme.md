# SelfDrivePyAuto

A Python-based autonomous driving system that uses computer vision and deep learning to control vehicle steering in simulation environments. The project implements a U-Net architecture with batch normalization for road detection and converts the detected path into steering commands.

## Features

- Real-time screen capture and processing for road detection
- Deep learning-based road segmentation using U-Net architecture
- Image annotation support using LabelMe for creating training data
- Automatic steering control using PyAutoGUI
- Training pipeline for custom road detection models
- Video to image conversion utilities for training data preparation
- Support for both CPU and GPU inference

## Requirements

- Python 3.6+
- PyTorch
- OpenCV
- NumPy
- PyAutoGUI
- Jupyter Notebook
- labelme

## Installation

```bash
# Clone the repository
git clone https://github.com/Indirajith-jithu/SelfDrivePyAuto.git
cd SelfDrivePyAuto

# Install required packages
pip install torch torchvision
pip install opencv-python
pip install numpy
pip install pyautogui
pip install jupyter
pip install labelme
```

## Project Structure

- `video_to_images.ipynb`: Utility for converting training videos to image frames
- `train_model.ipynb`: Training pipeline for the road detection model
- `predict_steering.ipynb`: Real-time prediction and steering control implementation

## Usage

1. **Data Preparation**
   - Use `video_to_images.ipynb` to convert your training videos into image frames
   - Organize your training data in appropriate directories

2. **Data Annotation**
   - Install and launch LabelMe: `labelme`
   - Open your image directory in LabelMe
   - Create polygons around the road areas
   - Save annotations in JSON format
   - Annotations will be used to create binary masks for training
   - Recommended annotation guidelines:
     * Label the drivable road area
     * Be consistent with the annotation boundaries
     * Include various road conditions and lighting
     * Annotate at least 50 images for basic training

3. **Model Training**
   - Open `train_model.ipynb` in Jupyter Notebook
   - Adjust hyperparameters if needed
   - Run all cells to train the model
   - The best model will be saved as 'best_model.pth'

4. **Autonomous Driving**
   - Open `predict_steering.ipynb` in Jupyter Notebook
   - Ensure your game/simulation window is properly positioned
   - Run the notebook to start autonomous driving
   - Press Ctrl+C in the notebook to stop the program

## Model Architecture

The project uses a U-Net architecture with the following features:
- Batch normalization for improved training stability
- Three encoder blocks and two decoder blocks
- Skip connections for better feature preservation
- Binary segmentation output for road detection

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

* PyTorch team for the deep learning framework
* OpenCV community for computer vision tools
* PyAutoGUI developers for system control capabilities

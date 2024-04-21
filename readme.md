# Neurocam

## Overview
Neurocam is an advanced surveillance system designed specifically for use with the Raspberry Pi 4 Compute Model and an IO board equipped with a PCI-e Akida neuromorphic processor. It leverages real-time person detection and generates detailed security reports using OpenAI's ChatGPT-4.

```
Security Report Overview:
The captured frame shows a residential garage with a dimly lit environment, primarily illuminated by a single bulb overhead. Visible contents include various household tools and a car parked inside. An individual is seen standing by the car, appearing to interact with the vehicle's door.

Persons Detected:
The frame includes one male individual, appearing in his mid-30s, with short dark hair and wearing casual attire consisting of a navy blue jacket and denim jeans. His posture and hand position near the car door handle suggest an attempt to gain entry into the vehicle.

Security Level: HIGH
```

The system runs the Visual Wake Word model for low power monitoring of a feed. When a person is detected, it switches to Yolo for person object tracking. Terminator vision is enabled when detecting the location of persons in the frame.

![Terminator Vision](https://i.imgur.com/XgUF3Cs.png)

## Features
- **Real-Time Person Detection**: Utilizes the Akida neuromorphic processor for efficient and rapid person detection.
- **Enhanced Security Reports**: Generates detailed security reports using ChatGPT-4, providing insights and alerts based on detected activities.
- **Optimized for Raspberry Pi**: Specifically designed to run on Raspberry Pi 4 with a Raspberry Pi Camera, integrating seamlessly with hardware capabilities.
- **Terminator Vision**: When a person is detected in frame it will switch to Terminator vision, the sweet red hue from the Terminator movies.

## Prerequisites
- Raspberry Pi 4 Compute Model with an IO Board
- PCI-e Akida Neuromorphic Processor ([link](https://shop.brainchipinc.com/products/akida%E2%84%A2-development-kit-pcie-board))>
- Raspberry Pi Camera Module
- Python 3.8 or higher
- A stable internet connection for setup and API interactions with ChatGPT

![Akida Neuromorphic SoC](https://i.imgur.com/g8YCnaX.jpeg)

## Installation

### Setup the Hardware
1. Connect the Raspberry Pi Camera to the Raspberry Pi 4 Compute Model.
2. Ensure the Akida Neuromorphic Processor is correctly installed in the PCI-e slot on the IO Board.

### Prepare the Software Environment
1. Create a virtual environment with access to system packages (required for `picamera2` module):
   ```bash
   python3 -m venv venv --system-site-packages
   source venv/bin/activate
   ```

2. Clone the repository:
   ```bash
    git clone https://github.com/stdp/neurocam.git
    cd neurocam
    ```

3. Install the required Python modules:
    ```bash
    pip install -r requirements.txt
    ```

4. You need to set up several environment variables that the system will use during operation, configure your `.env` file:

Copy the contents of `.env.example` to a new file named `.env` in the same directory:
   ```bash
   cp .env.example .env
   ```

## Usage
To start the Neurocam system, ensure your virtual environment is activated and execute:

``` bash
python neurocam.py
```

This will initiate the surveillance system, utilizing both the camera and the Akida neuromorphic processor for real-time detection and reporting.


## Contributing
Contributions are welcome! Feel free to fork the repository, make improvements, and submit pull requests. Whether it's bug fixes, feature additions, or documentation improvements, all contributions are appreciated.

# Neurocam

## Overview
Neurocam is an advanced surveillance system designed specifically for use with the Raspberry Pi 4 Compute Model and an IO board equipped with a PCI-e Akida neuromorphic processor. It leverages real-time person detection or face detection and generates detailed security reports using OpenAI's ChatGPT-4.


```
Security Report Overview:
The captured frame shows a residential garage with a dimly lit environment, primarily illuminated by a single bulb overhead. Visible contents include various household tools and a car parked inside. An individual is seen standing by the car, appearing to interact with the vehicle's door.

Persons Detected:
The frame includes one male individual, appearing in his mid-30s, with short dark hair and wearing casual attire consisting of a navy blue jacket and denim jeans. His posture and hand position near the car door handle suggest an attempt to gain entry into the vehicle.

Security Level: HIGH
```

The system runs the Visual Wake Word model for low power monitoring of a feed. When a person is detected, it switches to Yolo for person object tracking. By default it will also estimate the age of each face that is extracted. Terminator vision is optional and enabled when detecting the location of persons in the frame. When the security level is set to high, the RGB LEDs are triggered and flash red and blue (epilepsy warning).

![Terminator Vision](https://i.imgur.com/XgUF3Cs.png)

## Features
- **Real-Time Person Detection**: Utilises the Akida neuromorphic processor for efficient and rapid person detection.
- **Real-Time Face Detection**: Utilises the Akida neuromorphic processor for efficient and rapid face detection. Age can be predicted for each extracted face.
- **Enhanced Security Reports**: Generates detailed security reports using ChatGPT-4, providing insights and alerts based on detected activities.
- **Optimized for Raspberry Pi**: Specifically designed to run on Raspberry Pi 4 with a Raspberry Pi Camera, integrating seamlessly with hardware capabilities.
- **Terminator Vision**: When a person is detected in frame it will switch to Terminator vision, the sweet red hue from the Terminator movies.
- **Flashing RGB LEDs**: When the security level is set to HIGH, it triggers a set of WS2812 compatible RGB LEDs to flash red and blue.

## Prerequisites
- Raspberry Pi 4 Compute Model with an IO Board
- PCI-e Akida Neuromorphic Processor [link](https://shop.brainchipinc.com/products/akida%E2%84%A2-development-kit-pcie-board)
- Raspberry Pi Camera Module
- WS2812 compatible RGB LEDs [link](https://core-electronics.com.au/neopixel-stick-8-x-ws2812-5050-rgb-led-with-integrated-drivers.html)
- Python 3.8 or higher
- A stable internet connection for setup and API interactions with ChatGPT

![Akida Neuromorphic SoC](https://i.imgur.com/g8YCnaX.jpeg)

![WS2812 compatible RGB LEDs](https://i.imgur.com/zg9xneM.png)

## Installation

### Setup the Hardware
1. Connect the Raspberry Pi Camera to the Raspberry Pi 4 Compute Model.
2. Ensure the Akida Neuromorphic Processor is correctly installed in the PCI-e slot on the IO Board and the drivers are installed. ([link to instructions])(https://brainchip.com/support-akida-pcie-board)
3. Conect the WS2812 compatible RGB LED's wiring to the 5v, GROUND and GPIO 18

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

4. You need to set up several environment variables that the system will use during operation, configure your `.env` file. Copy the contents of `.env.example` to a new file named `.env` in the same directory:
   ```bash
   cp .env.example .env
   ```

## Usage

To start the Neurocam system, ensure your virtual environment is activated and follow the steps below. Python must be run as sudo for the LEDs to function:

1. get your virtualenv python path
``` bash
which python

# example output "/home/neuro/projects/neurocam/venv/bin/python"
```

2. copy the output of this and run the follow command, replacing python path with the output from the previous step:
```bash
sudo <python path> neurocam.py

# example command "sudo /home/neuro/projects/neurocam/venv/bin/python neurocam.py"
```

This will initiate the surveillance system, utilizing both the camera and the Akida neuromorphic processor for real-time detection and reporting.


## Contributing
Contributions are welcome! Feel free to fork the repository, make improvements, and submit pull requests. Whether it's bug fixes, feature additions, or documentation improvements, all contributions are appreciated.

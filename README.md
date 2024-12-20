# Multimodal Robot Navigation System

## Project Overview

This project aims to develop an multimodal robot navigation system that processes data from various sensors (such as IMU, camera, and LiDAR) through a perception module, a cross-domain attention module, and a decision-making module, ultimately outputting the robot's action strategy in real-world scenarios. The system is based on the CARLA simulator and utilizes deep learning techniques for training and testing.
![alt text](mmap.jpg)
## Table of Contents

- [Project Overview](#project-overview)
- [Environment Setup](#environment-setup)
- [How to Run the Code](#how-to-run-the-code)
- [Training and Testing](#training-and-testing)
- [Model Deployment](#Model-Deployment)

## Environment Setup

Before running the code, make sure your environment is configured with the following dependencies:

- Python 3.7+
- PyTorch 1.7+
- CARLA 0.9.11
- Other dependencies can be found in `requirements.txt`

### Install Dependencies

First, clone the project repository:

```bash
git clone https://github.com/yourusername/robot_navigation_system.git
cd robot_navigation_system
```
Then, install the Python dependencies:

```bash
pip install -r requirements.txt
```
note: Ensure that the CARLA simulator is installed and that the environment variables are correctly configured. You can refer to the CARLA official documentation for installation and configuration: https://carla.readthedocs.io/en/latest/build_linux/

## How to Run the Code
Run the Simulation
You can run the integrated system and test it in the CARLA simulator using the following command:

```bash
python run_simulation.py
```
This script will start the CARLA simulator and execute the complete perception, attention, and decision-making modules to generate the robot's navigation strategy.

## Training and Testing
Train the Model
Use the following command to train the model:
```bash
python main.py --mode train
```
This command will load data from the simulation dataset and train the perception, cross-domain attention, and decision-making modules. Training parameters and other hyperparameters can be configured in main.py.

Test the Model
After training is complete, you can test the model using the following command:

```bash
python main.py --mode test
```
This command will load the test dataset and evaluate the model's performance on new data, outputting the test loss.

## Model Deployment
It should be noted that the relevant underlying control configuration of our intelligent vehicle will be released later. Here, we only pre-release the relevant steps of algorithm deployment and reasoning.

Model training and derivation based on Jetson_robot：Before you can reason, you need to train the model and export it to ONNX format on a jetson robot：

```bash
python /_agent/_lightweight/train.py
```

Load the model and reason:

```bash
python _jetson_robot/deploy.py --onnx_model_path /_model/mmap_model.onnx
```


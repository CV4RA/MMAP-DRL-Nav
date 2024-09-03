# Integrated Robot Navigation System

## Project Overview

This project aims to develop an integrated robot navigation system that processes data from various sensors (such as IMU, camera, and LiDAR) through a perception module, a cross-domain attention module, and a decision-making module, ultimately outputting the robot's action strategy in real-world scenarios. The system is based on the CARLA simulator and utilizes deep learning techniques for training and testing.

## Table of Contents

- [Project Overview](#project-overview)
- [Environment Setup](#environment-setup)
- [How to Run the Code](#how-to-run-the-code)
- [Training and Testing](#training-and-testing)
- [Project Structure](#project-structure)
- [Download Links](#download-links)
- [Contribution Guidelines](#contribution-guidelines)
- [License](#license)

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
note: Ensure that the CARLA simulator is installed and that the environment variables are correctly configured. You can refer to the CARLA official documentation for installation and configuration: https://carla.readthedocs.io/en/latest/build_windows/

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

## Project Structure:
.
├── data/                       # Folder for storing datasets
├── models/                     # Contains code for perception, attention, and decision-making modules
│   ├── perception_module.py    # Implementation of the perception module
│   ├── attention_module.py     # Implementation of the cross-domain attention module
│   └── decision_module.py      # Implementation of the decision-making module
├── utils/                      # Contains utilities such as data loaders, loss functions, etc.
├── run_simulation.py           # Script to run the integrated system simulation
├── main.py                     # Main script for training and testing
├── requirements.txt            # List of dependencies
└── README.md                   # Project documentation

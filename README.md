# Vision-based robotic arm Project
Video of [simple pick-place task](./media/simple%20pick-and-place%20darts.mp4) and [complicated pick-place task](./media/complicated%20pick-and-place%20darts.mp4)
- [Vision-based robotic arm Project](#vision-based-robotic-arm-project)
  - [Overview](#overview)
    - [1. 6-DOF Robotic Arm Mechanical Structure Assembly](#1-6-dof-robotic-arm-mechanical-structure-assembly)
    - [2. Multi-DOF Robotic Arm Challenge Task](#2-multi-dof-robotic-arm-challenge-task)
    - [Provided Hardware:](#provided-hardware)
      - [Task Objective:](#task-objective)
    - [Requirements:](#requirements)
      - [Evaluation Criteria:](#evaluation-criteria)
    - [3. Dart Starting Point Identification using Image Recognition](#3-dart-starting-point-identification-using-image-recognition)
      - [Challenge:](#challenge)
      - [Solution Approach:](#solution-approach)
    - [4. Inverse Kinematics Analysis](#4-inverse-kinematics-analysis)
      - [Objectives:](#objectives)
  - [Features](#features)
  - [Directory Structure](#directory-structure)
  - [Software Components](#software-components)
    - [Embedded (Motor Control)](#embedded-motor-control)
    - [Vision (Object Detection)](#vision-object-detection)
  - [Setup and Usage](#setup-and-usage)

## Overview

### 1. 6-DOF Robotic Arm Mechanical Structure Assembly

### 2. Multi-DOF Robotic Arm Challenge Task

### Provided Hardware:
*   STM32F103 Development Board
*   Motor Driver Board
*   6-DOF Robotic Arm
*   Servos (must use the provided ones)

#### Task Objective:
Autonomously find, identify, and place magnetic darts in a specific order and position using the robotic arm.

### Requirements:
*   **Sensing (Optional but Recommended):** You can add sensors like cameras, color recognition sensors, ultrasonic sensors, etc., to aid in autonomously finding and identifying the magnetic darts.
*   **Dart Placement:** Pick up magnetic darts and place them on a target area (e.g., a dartboard).
*   **Adaptability:** The system must be able to handle variations in the initial positions of the magnetic darts without requiring code modifications.
*   **Manual Interaction:** Manual control methods (e.g., remote control) are permitted.
*   **Customization:** Custom mechanical parts for the arm structure can be designed and fabricated (3D printing only). Standard fasteners and structural parts are allowed; non-standard structures must be 3D printed.
*   **Dart Starting Positions:** Within the arm's reachable range, the initial positions of the magnetic darts can be placed arbitrarily (the 3 starting points must be different for evaluation). Use sensors (camera, ultrasonic, color sensor) to identify the initial position and color of the darts.
*   **Dart Ending Positions:** Pick up the darts with the manipulator and place them autonomously on the target area. The 3 final positions for the darts must be different.
*   **Control:** Control the robotic arm using serial communication to complete the task.

#### Evaluation Criteria:
Performance will be evaluated based on:
*   **Speed:** Time taken from the start of the arm's motor movement until all 3 darts are placed. Shorter times score higher.
*   **Placement Accuracy:** Precision of dart placement in the target locations.
*   **Identification Accuracy:** Correct identification of dart properties (if applicable, e.g., color).
*   **Smoothness:** Smoother arm movements result in higher scores.

### 3. Dart Starting Point Identification using Image Recognition
![darts' positions](/object_detection/media/1.jpg)
#### Challenge:
The provided dartboard has multiple colors similar to the darts, making visual identification difficult.

#### Solution Approach:
*   Cover the dartboard with white paper to simplify image recognition.
*   Image pre-processing using drawing tools can be applied if necessary.
*   Utilize MATLAB software and image processing techniques to identify the starting and ending positions of the darts.
*   Convert image coordinates to real-world coordinates using a pre-defined scale factor.

### 4. Inverse Kinematics Analysis

#### Objectives:
*   **Verification:** Use MATLAB to verify the correctness of the inverse kinematics (IK) formulas.
*   **Error Calculation:** Calculate the computational error associated with the IK formulas.
*   **Workspace Determination:** Determine the solvable range of the IK formulas to define the robotic arm's grasping range (workspace).
*   **Implementation:**
    *   Implement the IK solution in KEIL software to calculate the required angles for the 6 servos for any target point within the arm's workspace.
    *   Test the implementation by sending target coordinates via the serial port. The development board should solve the IK, control the arm to grasp the dart at the specified point, and thus verify the program's correctness.system.

## Features

*   **Motor Control:** Controls multiple servo motors (likely 6) for robotic arm movement using PWM signals generated by the STM32. Implements inverse kinematics for coordinate-based control.
*   **Object Detection:** Uses Python scripts for detecting objects, potentially based on color.
*   **Communication:** Likely uses UART serial communication to link the object detection results with the motor control actions.

## Directory Structure

*   `motor_control/`: Contains the embedded C code for the STM32 microcontroller.
    *   `Core/`: Main application code ([`main.c`](motor_control/Core/Src/main.c)) and header files ([`main.h`](motor_control/Core/Inc/main.h)).
    *   `Drivers/`: STM32 HAL library and CMSIS core files.
    *   `MDK-ARM/`: Keil MDK project files (`.uvprojx`, build outputs) and custom libraries (`Mylib/`).
        *   `Mylib/`: Contains custom motor control logic ([`motor.c`](motor_control/MDK-ARM/Mylib/motor.c), [`motor.h`](motor_control/MDK-ARM/Mylib/motor.h)) and UART handling ([`uart.c`](motor_control/MDK-ARM/Mylib/uart.c), [`uart.h`](motor_control/MDK-ARM/Mylib/uart.h)).
    *   `Indusrtial_controller.ioc`: STM32CubeMX configuration file.
*   `object_detection/`: Contains Python scripts for vision processing.
    *   `scripts/`: Python code for detection ([`color_detection.py`](object_detection/scripts/color_detection.py)), communication ([`com.py`](object_detection/scripts/com.py)), and main logic ([`final.py`](object_detection/scripts/final.py)).
    *   `media/`: Image files, possibly for testing or examples.
    *   `docs/`: Project documentation.

## Software Components

### Embedded (Motor Control)

*   **MCU:** STM32F103RB (Based on `Indusrtial_controller.ioc`)
*   **Language:** C
*   **IDE:** Keil MDK-ARM
*   **Libraries:** STM32 HAL, CMSIS
*   **Key Files:**
    *   [`motor_control/Core/Src/main.c`](motor_control/Core/Src/main.c): Main application loop, initialization.
    *   [`motor_control/MDK-ARM/Mylib/motor.c`](motor_control/MDK-ARM/Mylib/motor.c): Implements motor control functions like [`Motor_Init`](motor_control/MDK-ARM/Mylib/motor.c), [`Motor_Control`](motor_control/MDK-ARM/Mylib/motor.c), [`Motion_Control`](motor_control/MDK-ARM/Mylib/motor.c), [`angle2duty`](motor_control/MDK-ARM/Mylib/motor.c), [`duty2angle`](motor_control/MDK-ARM/Mylib/motor.c). Manages motor state variables ([`motor1`](motor_control/MDK-ARM/Mylib/motor.h) to [`motor6`](motor_control/MDK-ARM/Mylib/motor.h)).
    *   [`motor_control/MDK-ARM/Mylib/uart.c`](motor_control/MDK-ARM/Mylib/uart.c): Handles UART communication setup and data reception/parsing.
    *   [`motor_control/Core/Src/stm32f1xx_it.c`](motor_control/Core/Src/stm32f1xx_it.c): Interrupt service routines (e.g., for TIM3 used in [`Motor_Control`](motor_control/MDK-ARM/Mylib/motor.c)).

### Vision (Object Detection)

*   **Language:** Python
*   **Key Scripts:**
    *   [`object_detection/scripts/color_detection.py`](object_detection/scripts/color_detection.py): Performs color-based object detection.
    *   [`object_detection/scripts/com.py`](object_detection/scripts/com.py): Handles serial communication with the STM32 board.
    *   [`object_detection/scripts/final.py`](object_detection/scripts/final.py): Integrates detection and communication for the final application logic.
    *   (Dependencies might include OpenCV, PySerial - check script imports)

## Setup and Usage

1.  **Embedded:**
    *   Open the Keil project file: [`motor_control/MDK-ARM/Indusrtial_controller.uvprojx`](motor_control/MDK-ARM/Indusrtial_controller.uvprojx).
    *   Build the project using Keil MDK.
    *   Flash the compiled code (`.hex` file in `motor_control/MDK-ARM/`) to the STM32F103RB board using ST-Link or another compatible programmer.
2.  **Vision:**
    *   Ensure Python is installed.
    *   Install necessary Python packages (e.g., `pip install opencv-python pyserial numpy`).
    *   Connect the host PC to the STM32 board via USB-to-Serial adapter.
    *   Run the main Python script, likely [`object_detection/scripts/final.py`](object_detection/scripts/final.py).
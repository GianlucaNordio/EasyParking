# EasyParking - Parking Lot Management

**Project for the Computer Vision course (Academic Year 2023-2024)**  
**Department of Information Engineering, University of Padua**

**Authors:**  
- Giovanni Cinel, Student ID: 2103373  
- Davide Molinaroli, Student ID: 2104284  
- Gianluca Nordio, Student ID: 2109314  

## Overview
EasyParking is a project developed for managing parking lots using computer vision techniques. The system processes sequences of parking lot images to detect, segment, and classify parking spots, as well as evaluate the performance of these tasks. Key performance metrics, such as Mean Average Precision (MAP) and Intersection over Union (IoU), are calculated to assess the accuracy of the system.

## Features
The project performs the following tasks:
1. **Loading Datasets:** Loads a sequence of base images and dataset sequences for processing.
2. **Parking Spot Detection:** Detects parking spots in the images and initializes parking spot data.
3. **Segmentation:** Segments the images to identify the cars present.
4. **Classification:** Classifies the segmented areas to determine the occupancy status of parking spots.
5. **Minimap Creation:** Creates minimaps that visually represent the parking spots.
6. **Performance Evaluation:** Calculates and displays MAP and IoU metrics to evaluate the system’s performance.
7. **Visualization:** Displays results, including images with bounding boxes, masks, and classified results, for both the base sequence and dataset sequences.

## Requirements
This project is built using C++ and the OpenCV library. Ensure you have the following:
- OpenCV 4.x or higher
- C++11 or higher

## Running the Program
To run the program, simply execute the `main` function without any command-line arguments. The program will process the dataset and display results step by step.
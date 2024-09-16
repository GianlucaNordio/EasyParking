#include <iostream>
#include <fstream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>

#include "parkingSpot.hpp"
#include "parkingSpotDetector.hpp"
#include "utils.hpp"
#include "segmentation.hpp"
#include "parser.hpp"
#include "classification.hpp"
#include "performanceMeasurement.hpp"
#include "minimap.hpp"

const int NUMBER_SEQUENCES = 5;
const std::string DATASET_PATH = "../dataset";
const int MINIMAP_ROWS = 300;
const int MINIMAP_COLS = 500;

/**
 * @brief Main function that performs parking spot detection, segmentation, classification, and performance evaluation on a dataset.
 *
 * This function executes the following steps:
 * 
 * 1. Loads a dataset of parking lot images and sequences.
 * 2. Detects parking spots in the base sequence.
 * 3. Performs segmentation on both the base sequence and dataset sequences.
 * 4. Classifies the segmented parking spots in both the base sequence and dataset sequences.
 * 5. Evaluates the performance metrics (Mean Average Precision and Mean Intersection over Union) for both the base sequence and the dataset.
 * 6. Displays the results, including images, bounding boxes, segmented masks, classified masks, and performance metrics.
 * 
 * @return int Returns 0 upon successful execution.
 */
int main() {  

// STEP 1: Load the dataset

    std::cout << "STEP 1:" << std::endl;

    // Read the images from the dataset
    std::vector<cv::Mat> baseSequence;
    loadBaseSequenceFrames(DATASET_PATH, baseSequence);

    std::vector<std::vector<cv::Mat>> dataset;
    loadSequencesFrames(DATASET_PATH, NUMBER_SEQUENCES, dataset);
    
// STEP 2: Detect parking spots

    std::cout << "STEP 2:" << std::endl;

    std::vector<ParkingSpot> parkingSpot;
    detectParkingSpots(baseSequence, parkingSpot); 

    // Create a copy of the parkingSpot vector for each immage in the dataset
    std::vector<std::vector<ParkingSpot>> baseSequenceParkingSpot;
    for(int i = 0; i < baseSequence.size(); i++) {
        baseSequenceParkingSpot.push_back(parkingSpot);
    }

    std::vector<std::vector<std::vector<ParkingSpot>>> datasetParkingSpot;
    for(int i = 0; i < NUMBER_SEQUENCES; i++) {
        std::vector<std::vector<ParkingSpot>> sequenceParkingSpot;
        for(int j = 0; j < dataset[i].size(); j++) {
            sequenceParkingSpot.push_back(parkingSpot);
        }
        datasetParkingSpot.push_back(sequenceParkingSpot);
    }

    std::cout << "Detected parking spots in the base sequence.\n";

// STEP 3: Perform segmentation

    std::cout << "STEP 3:" << std::endl;

    std::vector<cv::Mat> baseSequenceMasks;
    std::vector<std::vector<cv::Mat>> datasetMasks;
    Segmentation segment(baseSequence);

    // Perform segmentation on the base sequence
    segment.segmentSequence(baseSequence, baseSequenceMasks);

    // Perform segmentation on the dataset
    for(int i = 0; i < NUMBER_SEQUENCES; i++) {
        std::vector<cv::Mat> sequenceMasks;
        segment.segmentSequence(dataset[i], sequenceMasks);
        datasetMasks.push_back(sequenceMasks);    
    }

    std::cout << "Segmented the base sequence and dataset sequences.\n";

// STEP 4: Perform classification

    std::cout << "STEP 4:" << std::endl;

    // Perform classification on the base sequence
    std::vector<cv::Mat> classifiedBaseSequenceMasks;
    classifySequence(baseSequenceParkingSpot, baseSequenceMasks, classifiedBaseSequenceMasks);

    // Perform classification on the dataset
    std::vector<std::vector<cv::Mat>> classifiedDatasetMasks;

    for(int i=0; i < NUMBER_SEQUENCES; i++) {
        std::vector<cv::Mat> classifiedSequenceMasks;
        classifySequence(datasetParkingSpot[i], datasetMasks[i], classifiedSequenceMasks);
        classifiedDatasetMasks.push_back(classifiedSequenceMasks);    
    }    
    
    std::cout << "Classified the base sequence and dataset sequences.\n";

// STEP 5: Minimap creation
    
    std::cout << "STEP 5:" << std::endl;

    std::vector<cv::Mat> baseSequenceMinimap;
    std::vector<std::vector<cv::Mat>> datasetMinimap;

    for(int i = 0; i < baseSequence.size(); i++)
        baseSequenceMinimap.push_back(cv::Mat(MINIMAP_ROWS, MINIMAP_COLS, CV_8UC3, cv::Scalar(255, 255, 255)));
        
    for(int i = 0; i < NUMBER_SEQUENCES; i++) {
        datasetMinimap.push_back(std::vector<cv::Mat>());
        for(int j = 0; j < dataset[i].size(); j++) {
            datasetMinimap[i].push_back(cv::Mat(MINIMAP_ROWS, MINIMAP_COLS, CV_8UC3, cv::Scalar(255, 255, 255)));
        }
    }

    // Create a minimap for the base sequence
    buildSequenceMinimap(baseSequenceParkingSpot, baseSequenceMinimap);

    // Create minimaps for the dataset
    for(int i = 0; i < NUMBER_SEQUENCES; i++) {
        buildSequenceMinimap(datasetParkingSpot[i], datasetMinimap[i]);
    }

    std::cout << "Created minimaps for the base sequence and dataset sequences.\n";


// STEP 6: Calculate performance metrics

    std::cout << "STEP 6:" << std::endl;

    std::vector<double> baseSequenceMAP;
    std::vector<double> baseSequenceIoU;

    double averageBaseSequenceMAP;
    double averageBaseSequenceIoU;

    std::vector<std::vector<double>> datasetMAP;
    std::vector<std::vector<double>> datasetIoU;

    std::vector<double> averageDatasetMAP;
    std::vector<double> averageDatasetIoU;
    

    performanceMeasurement(DATASET_PATH, NUMBER_SEQUENCES, parkingSpot, baseSequence, dataset, classifiedDatasetMasks, 
        classifiedBaseSequenceMasks, baseSequenceMAP, baseSequenceIoU, averageBaseSequenceMAP, averageBaseSequenceIoU, 
        datasetMAP, datasetIoU, averageDatasetMAP, averageDatasetIoU);

    std::cout << "Calculated performance metrics.\n";

// STEP 7: Display results

    std::cout << "STEP 7:" << std::endl;
    
    // Display the visual results

    // Display the base sequence
    addMinimap(baseSequenceMinimap, baseSequence);
    cv::imshow("Base sequence", produceSingleImage(baseSequence, 3));
    cv::waitKey();
    cv::destroyWindow("Base sequence");

    // Display the bounding boxes found in the base sequence
    std::vector<cv::Mat> baseSequenceBBoxes;
    printParkingSpot(parkingSpot, baseSequence, baseSequenceBBoxes);
    addMinimap(baseSequenceMinimap, baseSequenceBBoxes);

    cv::imshow("Base Sequence BBoxes", produceSingleImage(baseSequenceBBoxes, 3));
    cv::waitKey();
    cv::destroyWindow("Base Sequence BBoxes");

    // Display the segmented masks for the base sequence
    cv::imshow("Base Sequence Masks", produceSingleImage(baseSequenceMasks, 3));
    cv::waitKey();
    cv::destroyWindow("Base Sequence Masks");

    // Display the classified masks for the base sequence
    std::vector<cv::Mat> classifiedBaseSequenceMasksBGR;
    std::vector<cv::Mat> classifiedBaseSequenceMasksBGRwMask;
    classifiedBaseSequenceMasksBGRwMask.resize(classifiedBaseSequenceMasksBGR.size());
    convertGreyMaskToBGR(classifiedBaseSequenceMasks, classifiedBaseSequenceMasksBGR);

    for(int i = 0; i < classifiedBaseSequenceMasksBGR.size(); i++) {
        classifiedBaseSequenceMasksBGRwMask.push_back(cv::Mat::zeros(baseSequence[i].size(), baseSequence[i].type()));
        cv::addWeighted(baseSequence[i], 0.6, classifiedBaseSequenceMasksBGR[i], 0.4, 0, classifiedBaseSequenceMasksBGRwMask[i]);
    }

    addMinimap(baseSequenceMinimap, classifiedBaseSequenceMasksBGRwMask);

    cv::imshow("Base Sequence Classified Masks", produceSingleImage(classifiedBaseSequenceMasksBGRwMask, 3));
    cv::waitKey();
    cv::destroyWindow("Base Sequence Classified Masks");

    // Convert the greyscale masks to BGR
    std::vector<std::vector<cv::Mat>> classifiedDatasetMasksBGR;
    convertGreyMasksToBGR(classifiedDatasetMasks, classifiedDatasetMasksBGR);

     
    std::vector<std::vector<cv::Mat>> classifiedDatasetMasksBGRwMask;
    classifiedDatasetMasksBGRwMask.resize(classifiedDatasetMasksBGR.size());
    for(int i = 0; i < classifiedDatasetMasksBGR.size(); i++) {
        classifiedDatasetMasksBGRwMask.push_back(std::vector<cv::Mat>());
        for(int j = 0; j < classifiedDatasetMasksBGR[i].size(); j++) {
            classifiedDatasetMasksBGRwMask[i].push_back(cv::Mat::zeros(dataset[i][j].size(), dataset[i][j].type()));
            cv::addWeighted(dataset[i][j], 0.6, classifiedDatasetMasksBGR[i][j], 0.4, 0, classifiedDatasetMasksBGRwMask[i][j]);
        }
    } 

    // Display the results on the dataset one sequence at a time
    for(int i = 0; i < NUMBER_SEQUENCES; i++) {
        // For the sequence i:

        // Display the image
        cv::imshow("Sequence " + std::to_string(i + 1), produceSingleImage(dataset[i], 3));
        cv::waitKey();
        cv::destroyWindow("Sequence " + std::to_string(i + 1));

        // Display the bounding boxes found in the dataset
        std::vector<cv::Mat> sequenceBBoxes;
        printParkingSpot(parkingSpot, dataset[i], sequenceBBoxes);

        cv::imshow("Sequence " + std::to_string(i + 1) + " BBoxes", produceSingleImage(sequenceBBoxes, 3));
        cv::waitKey();
        cv::destroyWindow("Sequence " + std::to_string(i + 1) + " BBoxes");

        // Display the segmented masks for the dataset
        cv::imshow("Sequence " + std::to_string(i + 1) + " Masks", produceSingleImage(datasetMasks[i], 3));
        cv::waitKey();
        cv::destroyWindow("Sequence " + std::to_string(i + 1) + " Masks");

        // Display the classified masks for the dataset
        cv::imshow("Sequence " + std::to_string(i + 1) + " Classified Masks", produceSingleImage(classifiedDatasetMasksBGRwMask[i], 3));
        cv::waitKey();
        cv::destroyWindow("Sequence " + std::to_string(i + 1) + " Classified Masks");
    }
    // Display the performance metrics

    // Print performance for the base sequence
    std::cout << "Base Sequence Performance:\n";
    printPerformanceMetrics(baseSequenceMAP, baseSequenceIoU);
    std::cout << "Average MAP: " << std::fixed << std::setprecision(4) << averageBaseSequenceMAP << std::endl;
    std::cout << "Average IoU: " << std::fixed << std::setprecision(4) << averageBaseSequenceIoU << std::endl;
    std::cout << std::string(36, '-') << std::endl;
    std::cout << std::endl;

    std::cout << std::string(36, '=') << std::endl;

    // Print performance for the dataset
    std::cout << "\nDataset Performance:\n";
    for (int i = 0; i < datasetMAP.size(); i++) {
        std::cout << "Sequence " << i + 1 << ":\n";
        printPerformanceMetrics(datasetMAP[i], datasetIoU[i]);
        std::cout << "Average MAP: " << std::fixed << std::setprecision(4) << averageDatasetMAP[i] << std::endl;
        std::cout << "Average IoU: " << std::fixed << std::setprecision(4) << averageDatasetIoU[i] << std::endl;
        std::cout << std::string(36, '-') << std::endl;
        std::cout << std::endl;
    }

    return 0;
}
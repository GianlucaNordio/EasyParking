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

const int NUMBER_SEQUENCES = 5;

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

    // Read the images from the dataset
    std::vector<cv::Mat> baseSequence;
    loadBaseSequenceFrames("../dataset", baseSequence);

    std::vector<std::vector<cv::Mat>> dataset;
    loadSequencesFrames("../dataset", NUMBER_SEQUENCES, dataset);
    
// STEP 2: Detect parking spots
    std::vector<ParkingSpot> parkingSpot;
    detectParkingSpots(baseSequence, parkingSpot); 

// STEP 3: Perform segmentation
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

// STEP 4: Perform classification

    // Perform classification on the base sequence
    std::vector<cv::Mat> classifiedBaseSequenceMasks;
    classifySequence(parkingSpot, baseSequenceMasks, classifiedBaseSequenceMasks);

    // Perform classification on the dataset
    std::vector<std::vector<cv::Mat>> classifiedDatasetMasks;

    for(int i=0; i < NUMBER_SEQUENCES; i++) {
        std::vector<cv::Mat> classifiedSequenceMasks;
        classifySequence(parkingSpot, datasetMasks[i], classifiedSequenceMasks);
        classifiedDatasetMasks.push_back(classifiedSequenceMasks);    
    }    
    
// STEP 5: Calculate performance metrics

    // Load the ground truth
    cv::Mat baseSequenceMaskGT = cv::Mat::zeros(baseSequence[0].size(), CV_8UC1);
    std::vector<ParkingSpot> baseSequenceParkingSpotGT;
    std::vector<std::vector<ParkingSpot>> datasetParkingSpotGT;
    std::vector<std::vector<cv::Mat>> sequenceMaskGTGray;
    std::vector<std::vector<cv::Mat>> sequenceMaskGTBGR;

    loadBaseSequenceGroundTruth("../dataset", baseSequenceParkingSpotGT);
    loadSequencesGroundTruth("../dataset", NUMBER_SEQUENCES, datasetParkingSpotGT);
    loadSequencesSegMasks("../dataset", NUMBER_SEQUENCES, sequenceMaskGTGray);

    // Compute performance for the base sequence
    std::vector<double> baseSequenceMAP;
    std::vector<double> baseSequenceIoU;

    double averageBaseSequenceMAP = 0;
    double averageBaseSequenceIoU = 0;
    
    for(int i = 0; i < baseSequence.size(); i++) {
        baseSequenceMAP.push_back(calculateMeanAveragePrecision(baseSequenceParkingSpotGT, parkingSpot));
        baseSequenceIoU.push_back(calculateMeanIntersectionOverUnion(classifiedBaseSequenceMasks[i], baseSequenceMaskGT));
    }

    for(int i = 0; i < baseSequenceMAP.size(); i++) {
        averageBaseSequenceMAP += baseSequenceMAP[i];
        averageBaseSequenceIoU += baseSequenceIoU[i];
    }

    averageBaseSequenceMAP /= baseSequenceMAP.size();
    averageBaseSequenceIoU /= baseSequenceIoU.size();

    // Compute performance for the dataset
    std::vector<std::vector<double>> datasetMAP;
    std::vector<std::vector<double>> datasetIoU;

    std::vector<double> averageDatasetMAP;
    std::vector<double> averageDatasetIoU;

    for(int i = 0; i < NUMBER_SEQUENCES; i++) {
        std::vector<double> sequenceMAP;
        std::vector<double> sequenceIoU;

        for(int j = 0; j < dataset[i].size(); j++) {
            sequenceMAP.push_back(calculateMeanAveragePrecision(datasetParkingSpotGT[i], parkingSpot));
            sequenceIoU.push_back(calculateMeanIntersectionOverUnion(classifiedDatasetMasks[i][j], sequenceMaskGTGray[i][j]));
        }

        datasetMAP.push_back(sequenceMAP);
        datasetIoU.push_back(sequenceIoU);
    }

    for(int i = 0; i < datasetMAP.size(); i++) {
        double averageSequenceMAP = 0;
        double averageSequenceIoU = 0;

        for(int j = 0; j < datasetMAP[i].size(); j++) {
            averageSequenceMAP += datasetMAP[i][j];
            averageSequenceIoU += datasetIoU[i][j];
        }

        averageSequenceMAP /= datasetMAP[i].size();
        averageSequenceIoU /= datasetIoU[i].size();

        averageDatasetMAP.push_back(averageSequenceMAP);
        averageDatasetIoU.push_back(averageSequenceIoU);
    }

// STEP 6: Display results
    
    // Display the visual results

    // Display the base sequence
    cv::imshow("Base sequence", produceSingleImage(baseSequence, 3));
    cv::waitKey();

    // Display the bounding boxes found in the base sequence
    std::vector<cv::Mat> baseSequenceBBoxes;
    for(int i = 0; i < baseSequence.size(); i++) {
        cv::Mat output = baseSequence[i].clone();
        for(int j = 0; j < parkingSpot.size(); j++) {
            cv::rectangle(output, parkingSpot[j].rect.boundingRect(), cv::Scalar(0, 255, 0), 2);
        }
        baseSequenceBBoxes.push_back(output);
    }
    cv::imshow("Base Sequence BBoxes", produceSingleImage(baseSequenceBBoxes, 3));
    cv::waitKey();

    // Display the segmented masks for the base sequence
    cv::imshow("Base Sequence Masks", produceSingleImage(baseSequenceMasks, 3));
    cv::waitKey();

    // Display the classified masks for the base sequence
    std::vector<cv::Mat> classifiedBaseSequenceMasksBGR;
    convertGreyMaskToBGR(classifiedBaseSequenceMasks, classifiedBaseSequenceMasksBGR);
    cv::imshow("Base Sequence Classified Masks", produceSingleImage(classifiedBaseSequenceMasksBGR, 3));
    cv::waitKey();

    // Convert the greyscale masks to BGR
    std::vector<std::vector<cv::Mat>> classifiedDatasetMasksBGR;
    convertGreyMaskToBGR(classifiedDatasetMasks, classifiedDatasetMasksBGR);

    // Display the results on the dataset one sequence at a time
    for(int i = 0; i < NUMBER_SEQUENCES; i++) {
        // For the sequence i:

        // Display the image
        cv::imshow("Sequence " + std::to_string(i + 1), produceSingleImage(dataset[i], 3));
        cv::waitKey();

        // Display the bounding boxes found in the dataset
        std::vector<cv::Mat> sequenceBBoxes;
        for(int j = 0; j < dataset[i].size(); j++) {
            cv::Mat output = dataset[i][j].clone();
            for(int k = 0; k < parkingSpot.size(); k++) {
                cv::rectangle(output, parkingSpot[k].rect.boundingRect(), cv::Scalar(0, 255, 0), 2);
            }
            sequenceBBoxes.push_back(output);
        }

        cv::imshow("Sequence " + std::to_string(i + 1) + " BBoxes", produceSingleImage(sequenceBBoxes, 3));
        cv::waitKey();

        // Display the segmented masks for the dataset
        cv::imshow("Sequence " + std::to_string(i + 1) + " Masks", produceSingleImage(datasetMasks[i], 3));
        cv::waitKey();

        // Display the classified masks for the dataset
        cv::imshow("Sequence " + std::to_string(i + 1) + " Classified Masks", produceSingleImage(classifiedDatasetMasksBGR[i], 3));
        cv::waitKey();
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
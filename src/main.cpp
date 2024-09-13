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

int main() {  

// STEP 1: Load the dataset

    // Read the images from the dataset
    std::vector<cv::Mat> baseSequence;
    loadBaseSequenceFrames("../dataset", baseSequence);
    cv::imshow("Base sequence", produceSingleImage(baseSequence, 3));
    cv::waitKey();

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

    // Display the performance metrics

    // Print performance for the base sequence
    std::cout << "Base Sequence Performance:\n";
    printPerformanceMetrics(baseSequenceMAP, baseSequenceIoU);
    std::cout << "Average MAP: " << std::fixed << std::setprecision(4) << averageBaseSequenceMAP << std::endl;
    std::cout << "Average IoU: " << std::fixed << std::setprecision(4) << averageBaseSequenceIoU << std::endl;

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
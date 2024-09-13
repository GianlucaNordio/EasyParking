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
    std::vector<ParkingSpot> baseSequenceParkingSpotGT;
    std::vector<std::vector<ParkingSpot>> datasetParkingSpotGT;
    std::vector<std::vector<cv::Mat>> sequenceMaskGTGray;
    std::vector<std::vector<cv::Mat>> sequenceMaskGTBGR;

    loadBaseSequenceGroundTruth("../dataset", baseSequenceParkingSpotGT);
    loadSequencesGroundTruth("../dataset", NUMBER_SEQUENCES, datasetParkingSpotGT);
    loadSequencesSegMasks("../dataset", NUMBER_SEQUENCES, sequenceMaskGTGray);

    // Convert gray masks to BGR
    convertGreyMaskToBGR(sequenceMaskGTGray, sequenceMaskGTBGR);

    // Compute performance for the base sequence
    
    

    // Compute performance for the dataset


    // STEP 6: Display results
    

    return 0;
}
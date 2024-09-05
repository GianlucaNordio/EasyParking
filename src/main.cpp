#include <iostream>
#include <fstream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp> // required to use the function cv::line
#include <filesystem>

#include "parkingSpot.hpp"
#include "parkingSpotDetector.hpp"
#include "utils.hpp"
#include "segmentation.hpp"

const int NUMBER_SEQUENCES = 5;

int main() {
     // Detect bounding boxes

        // Read sequence 0 to use images to detect parking spots
        // Find parking spots for each image
        // Non maxima suppression to remove overlapping bounding boxes
        // Build the 2D map of parking spots
    
    
    // Read the images from the dataset
    std::vector<cv::Mat> images;
    loadBaseSequenceFrames("../dataset", images);
    cv::imshow("Base sequence", produceSingleImage(images, 3));
    cv::waitKey();


    // Call the function to detect parking spots
    
    /*
    std::vector<ParkingSpot> parkingSpot;
    detectParkingSpot(images, parkingSpot); 
    */


    // Load the other frames relative to the test sequences
    std::vector<std::vector<cv::Mat>> data;
    loadSequencesFrames("../dataset", NUMBER_SEQUENCES, data);
    
    for(int i = 0; i < data.size(); i++) {
        cv::imshow("Test Data", produceSingleImage(data[i], 3));
        cv::waitKey();
    }
    
    
  
    // Classify parking spots

        // Read sequence 1-5 to use images to classify parking spots
        // Classify parking spots for each image
        // Draw bounding boxes and classification on the image
    
    // Segment car in the images
    Segmentation segm(images);
    std::vector<cv::Mat> masks;
    segm.segmentVectorImages(data[0], masks);
    cv::imshow("Test Data", produceSingleImage(masks, 3));
    cv::waitKey();

    // Dummy code simply to show the correct segmentation masks
    /*
    std::vector<std::vector<cv::Mat>> groundTruthMasksGray;
    loadSequencesSegMasks("../dataset", NUMBER_SEQUENCES, groundTruthMasksGray);
    std::vector<std::vector<cv::Mat>> groundTruthMasksBGR = groundTruthMasksGray;
    convertGreyMaskToBGR(groundTruthMasksGray, groundTruthMasksBGR);
    for(int i = 0; i < groundTruthMasksBGR.size(); i++) {
        cv::Mat test = produceSingleImage(groundTruthMasksBGR[i], 3);
        cv::imshow("Test Data", test);
        cv::waitKey();
    }
    */


    // Performance measure

    return 0;
}
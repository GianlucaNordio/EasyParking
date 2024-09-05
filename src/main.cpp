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
    
    std::vector<ParkingSpot> parkingSpot;
    detectParkingSpots(images, parkingSpot); 
    

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
    
    // test(images,data[4][0]);

    // Performance measure

    return 0;
}
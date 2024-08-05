#include <iostream>
#include <fstream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp> // required to use the function cv::line
#include <filesystem>
#include "parkingSpot/parkingSpot.hpp"
#include "parkingSpotDetector/parkingSpotDetector.hpp"

int main() {
    
    // Read the images from the dataset
    std::string folderPath = "../dataset/sequence0/frames";
    std::vector<cv::Mat> images;

    for(const auto& entry : std::filesystem::directory_iterator(folderPath)) {
        if(entry.is_regular_file()){
            std::string filePath = entry.path().string();
            cv::Mat image = cv::imread(filePath);
            if(!image.empty()){
                images.push_back(image);
                std::cout << "Read image: " << filePath << std::endl;
            } else
                std::cerr << "Could not read image: " << filePath << std::endl;
        }
    }

    // Call the function to detect parking spots

    std::vector<ParkingSpot> parkingSpot;
    detectParkingSpot(images, parkingSpot);

    // Detect bounding boxes

        // Read sequence 0 to use images to detect parking spots
        // Find parking spots for each image
        // Non maxima suppression to remove overlapping bounding boxes
        // Build the 2D map of parking spots
    
    // Classify parking spots

        // Read sequence 1-5 to use images to classify parking spots
        // Classify parking spots for each image
        // Draw bounding boxes and classification on the image
    
    // Segment car in the images


    // Performance measure

    return 0;
}
#include <iostream>
#include <fstream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp> // required to use the function cv::line
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

    std::vector<std::vector<cv::Mat>> data;
    loadSequencesFrames("../dataset", NUMBER_SEQUENCES, data);
    
    std::vector<std::vector<cv::Mat>> allMasks;
    // Perform segmentation
    Segmentation segment(images);
    for(int i = 0; i < NUMBER_SEQUENCES; i++) {
        std::vector<cv::Mat> masks;
        //cv::imshow("Original Data", produceSingleImage(data[i], 3));
        segment.segmentVectorImages(data[i], masks);
        allMasks.push_back(masks);
        cv::imshow("MASK TEST", masks[0]);
        cv::waitKey();
        allMasks.push_back(masks);    
        //cv::imshow("Test Data", produceSingleImage(masks, 3));
        //cv::waitKey();
    }
    std::vector<ParkingSpot> spaces = parseXML("../dataset/sequence0/bounding_boxes/2013-02-24_10_05_04.xml");
    cv::Mat image = cv::imread("../dataset/sequence1/masks/2013-02-22_06_25_00.png");
    cv::cvtColor(image,image, cv::COLOR_BGR2GRAY);
    std::vector<cv::Mat> ciao;
    ciao.push_back(classifyCars(spaces, allMasks[0][1]));
    std::vector<cv::Mat> image2;
    image2.push_back(image);
    std::cout<< "Test segmentation metric:" << calculateMeanIntersectionOverUnion(ciao , image2) << std::endl;


   // for(int i = 0; i < NUMBER_SEQUENCES; i++) {
        
//    }
    cv::imshow("boh", classifyCars(spaces, allMasks[0][1]));

    // Call the function to detect parking spots
    std::vector<ParkingSpot> parkingSpot;
    detectParkingSpots(images, parkingSpot); 

    // Perform classification of the parking spot

    
    



    /*
    // Load the other frames relative to the test sequences
    std::vector<std::vector<cv::Mat>> data;
    loadSequencesFrames("../dataset", NUMBER_SEQUENCES, data);
    
    for(int i = 0; i < data.size(); i++) {
        cv::imshow("Test Data", produceSingleImage(data[i], 3));
        cv::waitKey();
    }
    */
    
    
  
    // Classify parking spots

        // Read sequence 1-5 to use images to classify parking spots
        // Classify parking spots for each image
        // Draw bounding boxes and classification on the image
    
    // Segment car in the images
    // Segmentation segm(images);
    // std::vector<cv::Mat> masks;
    // segm.segmentVectorImages(data[3], masks);
    // cv::imshow("Test Data", produceSingleImage(masks, 3));
    // cv::waitKey();
    
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
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp> 
#include <iostream>

#include "utils.hpp"


cv::Mat produceSingleImage(const std::vector<cv::Mat>& images, int imagesPerLine) {
    // TODO understand if the exceptions I throw are okay
    
    // Check if at least one  image was provided
    if(images.size() < 1) {
        throw std::invalid_argument("No image was provided");
    }
    
    // Check if all images have the same size
    int requiredCols = images[0].cols;
    int requiredRows = images[0].rows;
    for(int i = 1; i < images.size(); i++) {
        if(requiredCols != images[i].cols) {
            throw std::invalid_argument("Images have different number of columns");
        }
        else if(requiredRows != images[i].rows) {
            throw std::invalid_argument("Images have different number of rows");
        }
    }
    // If total number of images smaller then images per line reduce the images per line
    if(imagesPerLine > images.size()) {
        imagesPerLine = images.size();
    }

    // Initialize the final image
    int imagesPerColumn = std::ceil(images.size() / static_cast<float>(imagesPerLine));
    int totalCols = images[0].cols * imagesPerLine;
    int totalRows = images[0].rows * imagesPerColumn;
    cv::Mat result = cv::Mat::zeros(cv::Size(totalCols, totalRows), images[0].type());
    
    // Copy the images into the final image
    for(int i = 0; i < images.size(); i++) {
        int x = i % imagesPerLine;
        int y = i / imagesPerLine;
        cv::Mat position(result,
                        cv::Range(images[i].rows * y, images[i].rows * y + images[i].rows),
                        cv::Range(images[i].cols * x, images[i].cols * x + images[i].cols)
                        );
        images[i].copyTo(position);
    }

    // TODO remove this resize (used only to be able to fit the image in our screen)
    cv::resize(result, result, cv::Size(result.cols/2, result.rows/2));

    return result;
}



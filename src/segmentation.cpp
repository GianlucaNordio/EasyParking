#include "segmentation.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/bgsegm.hpp>

/**
 * Constructor for the Segmentation class.
 * Initializes a background subtractor (MOG2) with a specified history length, variance threshold, and shade detection setting.
 * It applies the background subtractor to the provided background images to initialize the model.
 *
 * @param backgroundImages A vector of cv::Mat objects representing the background images used to initialize the background subtractor.
 */
Segmentation::Segmentation(const std::vector<cv::Mat> &backgroundImages) {
    pBackSub = cv::bgsegm::createBackgroundSubtractorGMG();
    
    cv::Mat mask;
    for(int i = 0; i < backgroundImages.size(); i++) {
        pBackSub -> apply(backgroundImages[i], mask);
    }

}

/**
* Segments a sequence of images by applying the segmentImage function to each image in the input vector.
* The results are stored in a vector of output masks corresponding to each image.
*
* @param images       A vector of cv::Mat objects representing the input images to be segmented.
* @param outputMasks  A reference to a vector of cv::Mat objects where the output masks for each image will be stored.
*/
void Segmentation::segmentSequence(const std::vector<cv::Mat> &images, std::vector<cv::Mat> &outputMasks) {
 
    // Apply segmentImage to each element of the vector separately
    for(int i = 0; i < images.size(); i++) {
        outputMasks.push_back(cv::Mat());
        segmentImage(images[i], outputMasks[i]);
    }
}

/**
 * Segments a single image using background subtraction and connected component analysis.
 * The method removes noise, filters small connected components, and applies morphological operations
 * to generate a refined output mask.
 *
 * @param image      A cv::Mat object representing the input image to be segmented.
 * @param outputMask A reference to a cv::Mat object where the output binary mask will be stored.
 */
void Segmentation::segmentImage(const cv::Mat &image, cv::Mat &outputMask) {
    
    pBackSub -> apply(image, outputMask, BACKGROUND_NOT_UPDATED);
    
    // Remove the shadow parts and the noise 
    //(tried using the noShade option in the constructor but still finds shades, simply doesn't differentiate them)
    cv::threshold(outputMask, outputMask, SHADOW_LOW_THRESHOLD, SHADOW_HIGH_THRESHOLD, cv::THRESH_BINARY);

    // Compute connected components and their statistics
    cv::Mat stats, centroids, labelImage;
    int numLabels = cv::connectedComponentsWithStats(outputMask, labelImage, stats, centroids, CONNECTIVITY_8, CV_32S);

    // Create a mask for large connected components
    cv::Mat mask = cv::Mat::zeros(labelImage.size(), IMAGE_TYPE_1_CHANNEL);

    // Filter components based on area size threshold
    for (int i = 1; i < numLabels; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area > PIXEL_SIZE_THRESHOLD) {
            mask = mask | (labelImage == i);
        }
    }

    // Perform morphological closing to refine the mask (remove white lines)
    cv::Mat element = cv::getStructuringElement(MORPH_RECT, 
        cv::Size(2 * MORPH_SIZE + 1, 2 * MORPH_SIZE + 1), 
        cv::Point(MORPH_SIZE, MORPH_SIZE));

    cv::Mat closedMask;
    cv::morphologyEx(mask, closedMask, cv::MORPH_CLOSE, element);

    // Mask the top right corner of the image as specified in the project assumptions
    maskRightTopCorner(closedMask);

    // Apply the refined mask to the input image
    outputMask = closedMask;
}

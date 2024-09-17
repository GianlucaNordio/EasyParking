// Gianluca Nordio

#ifndef SEGMENTATION_HPP
#define SEGMENTATION_HPP

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


#include "utils.hpp"
#include "constants.hpp"

/**
 * @brief Class that allows to perform segmentation based on background subtraction.
 * To work we first need to call the constructor with the set of images about the background.
 * Then we can simply perform background substraction.
 */
class Segmentation {
    private:
        // Pointer containing the background subtractor
        cv::Ptr<cv::BackgroundSubtractor> pBackSub;

    public:

        /**
        * Constructor for the Segmentation class.
        * Initializes a background subtractor (MOG2) with a specified history length, variance threshold, and shade detection setting.
        * It applies the background subtractor to the provided background images to initialize the model.
        *
        * @param backgroundImages A vector of cv::Mat objects representing the background images used to initialize the background subtractor.
        */
        Segmentation(const std::vector<cv::Mat> &backgroundImages);

        /**
        * Segments a sequence of images by applying the segmentImage function to each image in the input vector.
        * The results are stored in a vector of output masks corresponding to each image.
        *
        * @param images       A vector of cv::Mat objects representing the input images to be segmented.
        * @param outputMasks  A reference to a vector of cv::Mat objects where the output masks for each image will be stored.
        */
        void segmentSequence(const std::vector<cv::Mat> &images, std::vector<cv::Mat> &outputMasks);
        
        /**
        * Segments a single image using background subtraction and connected component analysis.
        * The method removes noise, filters small connected components, and applies morphological operations
        * to generate a refined output mask.
        *
        * @param image      A cv::Mat object representing the input image to be segmented.
        * @param outputMask A reference to a cv::Mat object where the output binary mask will be stored.
        */
        void segmentImage(const cv::Mat &image, cv::Mat &outputMask);
};


#endif // SEGMENTATION_HP
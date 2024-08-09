#ifndef SEGMENTATION_HPP
#define SEGMENTATION_HPP

#include <opencv2/opencv.hpp>

/**
 * Class that allows to perform segmentation based on background subtraction.
 * To work we first need to call the constructor with the set of images about the background.
 * Then we can simply perform background substraction.
 */
class Segmentation {
    private:
        cv::Ptr<cv::BackgroundSubtractor> pBackSub;
        const int BACKGROUND_NOT_UPDATED = 0;

    public:
        Segmentation(const std::vector<cv::Mat> &backgroundImages);

        /** 
         * @brief perform segmentation on a single image
         * @param image the image on which segmentation should be performed
         * @param outputMask the mask on which we should store the background substraction
         */
        void segmentImage(const cv::Mat &image, cv::Mat &outputMask);
        void segmentVectorImages(const std::vector<cv::Mat> &images, std::vector<cv::Mat> &outputMasks);
};


#endif // SEGMENTATION_HP
#ifndef SEGMENTATION_HPP
#define SEGMENTATION_HPP

#include <opencv2/opencv.hpp>

enum labelId
{
    background,
    carInsideParkingSpot,
    carOutsideParkingSpot
};

/**
 * @brief Class that allows to perform segmentation based on background subtraction.
 * To work we first need to call the constructor with the set of images about the background.
 * Then we can simply perform background substraction.
 */
class Segmentation {
    private:
        // Pointer containing the background subtractor
        cv::Ptr<cv::BackgroundSubtractor> pBackSub;
        // Parameter allowing the background subtractor to not update the background model
        const int BACKGROUND_NOT_UPDATED = 0;

    public:
        Segmentation(const std::vector<cv::Mat> &backgroundImages);

        /** 
         * @brief perform segmentation on a single image
         * @param image the image on which segmentation should be performed
         * @param outputMask the mask on which we should store the background substraction
         */
        void segmentImage(const cv::Mat &image, cv::Mat &outputMask);

        /** 
         * @brief perform segmentation on a multiple images
         * @param images the images on which segmentation should be performed
         * @param outputMasks the masks on which we should store the background substraction
         */
        void segmentVectorImages(const std::vector<cv::Mat> &images, std::vector<cv::Mat> &outputMasks);
};


#endif // SEGMENTATION_HP
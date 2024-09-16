#include "classification.hpp"


/**
 * Classifies a sequence of images by processing segmentation masks and stores the classified results in the output vector.
 *
 * @param parkingSpot            A vector of vector of ParkingSpot objects representing the parking spaces.
 * @param segmentationMasks      A vector of cv::Mat representing the segmentation masks for each image.
 * @param classifiedMasks        A reference to a vector of cv::Mat where the classified results will be stored.
 */
void classifySequence(std::vector<std::vector<ParkingSpot>>& parkingSpot, std::vector<cv::Mat> segmentationMasks, std::vector<cv::Mat>& classifiedMasks) {
    for (int i = 0; i < segmentationMasks.size(); i++) {
        cv::Mat classifiedMask;
        classifyImage(parkingSpot[i], segmentationMasks[i], classifiedMask);
        classifiedMasks.push_back(classifiedMask);
    }
}

/**
 * @brief Classifies parking spots in an image based on segmentation mask.
 *
 * This function processes an input segmentation mask to classify parking spots 
 * as either occupied or free. It utilizes connected components analysis to identify
 * distinct components in the segmentation mask and then checks each parking spot 
 * against these components. The classification is based on the percentage of each 
 * component that overlaps with the parking spot.
 *
 * @param[in] parkingSpot        A vector of `ParkingSpot` objects, each representing a 
 *                                 parking spot with its associated rectangle.
 * @param[in] segmentationMask   A binary mask where connected components are 
 *                                 identified and labeled.
 * @param[out] classifiedMask    An output mask of the same size as the 
 *                                 segmentationMask, where each pixel value indicates 
 *                                 the classification of the component (0: background, 
 *                                 1: car inside, 2: car outside).
 *
 * @note The `PERCENTAGE_INSIDE_THRESHOLD` is used to determine if a parking spot 
 *       is considered occupied based on the percentage of the component's area 
 *       covered by the parking spot.
 */
void classifyImage(std::vector<ParkingSpot>& parkingSpot, cv::Mat segmentationMask, cv::Mat& classifiedMask) {
    cv::Mat labels, stats, centroids;
    int numComponents = cv::connectedComponentsWithStats(segmentationMask, labels, stats, centroids);

    classifiedMask = cv::Mat::zeros(segmentationMask.size(), IMAGE_TYPE_1_CANALE); // Initialize with zeros

    for(int j = 1; j < numComponents; j++){
        // Set as outside parking spot all the components that are not the background
        classifiedMask.setTo(labelId::carOutsideParkingSpot, labels == j);
    }

    // Create a mask for each parking spot
    for (auto& spot : parkingSpot) {
        cv::Mat spotMask = cv::Mat::zeros(segmentationMask.size(), IMAGE_TYPE_1_CANALE);
        cv::Point2f vertices[4];
        spot.rect.points(vertices);
        std::vector<cv::Point> contour(vertices, vertices + 4);
        cv::fillConvexPoly(spotMask, contour, WHITE_ONE_CHANNEL);

        for (int j = 1; j < numComponents; j++) {
            cv::Mat componentMask = (labels == j);
            cv::Mat intersectionMask;
            cv::bitwise_and(spotMask, componentMask, intersectionMask);

            int componentArea = stats.at<int>(j, cv::CC_STAT_AREA);
            int insidePixels = cv::countNonZero(intersectionMask);
            double percentageInside = (static_cast<double>(insidePixels) / componentArea) * PERCENTAGE;

            // Update classifiedMask based on percentage inside
            if (percentageInside > PERCENTAGE_INSIDE_THRESHOLD) {
                classifiedMask.setTo(labelId::carInsideParkingSpot, componentMask);
                spot.occupied = true;
            }
        }
    }
}
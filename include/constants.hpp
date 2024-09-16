#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

#include <opencv2/opencv.hpp>

/**
 * @brief Separator used for dividing sections in the console output (type 1).
 */
const char SEPARATOR_TYPE_1 = '-';

/**
 * @brief Separator used for dividing sections in the console output (type 2).
 */
const char SEPARATOR_TYPE_2 = '=';

/**
 * @brief Path to the dataset containing the parking lot images and sequences.
 */
const std::string DATASET_PATH = "../dataset";

/**
 * @brief The name of the directory containing segmentation masks.
 */
const std::string MASKS_FOLDER = "masks";

/**
 * @brief The name of the directory containing bounding box annotations for ground truth.
 */
const std::string BOUNDING_BOX_FOLDER = "bounding_boxes";

/**
 * @brief The name of the directory containing the parking images.
 */
const std::string FRAMES_FOLDER = "frames";

/**
 * The prefix used for sequence directories.
 */
const std::string SEQUENCE = "sequence";

/**
 * @brief The directory separator character used in paths.
 */
const std::string SLASH = "/";

/**
 * @brief Color white used for initializing empty images and minimaps.
 */
const cv::Scalar WHITE = cv::Scalar(255, 255, 255);

/**
 * @brief Color red used for drawing bounding boxes.
 */
const cv::Scalar RED = cv::Scalar(0, 0, 255);

/**
 * @brief Color blue used for drawing parking spots.
 */
const cv::Scalar BLUE = cv::Scalar(255, 0, 0);

/**
 * @brief Size of the rectangle used for drawing bounding boxes on the minimap.
 */
const cv::Size SIZE_RECT_MINIMAP = cv::Size(60,20);

/**
 * @brief Weight for classified images in the blend between original and classified images.
 */
const double CLASSIFIED_IMAGE_WEIGHT = 0.4;

/**
 * @brief Weight for original images in the blend between original and classified images.
 */
const double ORIGINAL_IMAGE_WEIGHT = 0.6;

/**
 * @brief A double value representing the overlap threshold. This value indicates the ratio of intersection area 
 *        between two rectangles relative to their total area. If the normalized intersection area between two 
 *        rectangles exceeds this threshold, one of them will be marked for removal.
 */
const double NON_MAXIMUM_SUPPRESSION_THRESHOLD = 0.3;

/**
 * @brief Threshold used for Intersection over Union (IoU) calculation.
 */
const double IOU_THRESHOLD = 0.5;

/**
 * @brief A constant threshold used to determine if car is inside a parking spot.
 *        If more than PERCENTAGE_INSIDE_THRESHOLD*100% of the component is inside the parking spot, it is classified as 'inside'.
 */
const float PERCENTAGE_INSIDE_THRESHOLD = 0.7;

/**
 * @brief Type of image (3-channel 8-bit color image).
 */
const int IMAGE_TYPE_3_CANALI = CV_8UC3;

/**
 * @brief Type of image (1-channel 8-bit color image).
 */
const int IMAGE_TYPE_1_CANALE = CV_8U;

/**
 * @brief Shift used for adjusting the color when combining images.
 */
const int SHIFT = 0;

/**
 * @brief The index of the base sequence.
 */
const int BASE_SEQUENCE_INDEX = 0;

/**
 * @brief Number of images to display per row when visualizing results.
 */
const int NUMBER_OF_IMAGES_FOR_ROW = 3;

/**
 * @brief Number of image sequences in the dataset.
 */
const int NUMBER_SEQUENCES = 5;

/**
 * @brief Thickness of the lines used for drawing bounding boxes.
 */
const int LINE_THICKNESS = 2;

/**
 * @brief Length of the separation line used for console output formatting.
 */
const int SEPARATION_LINE_LENGTH = 36;

/**
 * @brief Threshold used for aligning rectangles.
 */
const double ALIGNED_RECTS_THRESHOLD = 30;

/**
 * @brief Value used for the black area in the mask in template matching.
 */
const int MASK_LINE_VALUE_LOW = 10;

/**
 * @brief Value used for the white area in the mask in template matching.
 */
const int MASK_LINE_VALUE_HIGH = 245;

/**
 * @brief Value used for the template in template matching.
 */
const int TEMPLATE_LINE_VALUE = 255;

/**
 * @brief Number of rows for the minimap image.
 */
const int MINIMAP_ROWS = 300;

/**
 * @brief Number of columns for the minimap image.
 */
const int MINIMAP_COLS = 500;

/**
 * @brief Height of the minimap.
 */
const int MAP_HEIGHT = 250;

/**
 * @brief Width of the minimap.
 */
const int MAP_WIDTH = 450;

#endif // CONSTANTS_HPP
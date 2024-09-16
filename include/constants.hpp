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
 * The name of the directory containing segmentation masks.
 */
const std::string MASKS_FOLDER = "masks";

/**
 * The name of the directory containing bounding box annotations for ground truth.
 */
const std::string BOUNDING_BOX_FOLDER = "bounding_boxes";

/**
 * The name of the directory containing the parking images.
 */
const std::string FRAMES_FOLDER = "frames";

/**
 * The prefix used for sequence directories.
 */
const std::string SEQUENCE = "sequence";

/**
 * The directory separator character used in paths.
 */
const std::string SLASH = "/";

/**
 * @brief Color white used for initializing empty images and minimaps.
 */
const cv::Scalar WHITE = cv::Scalar(255, 255, 255);

/**
 * @brief Weight for classified images in the blend between original and classified images.
 */
const double CLASSIFIED_IMAGE_WEIGHT = 0.4;

/**
 * @brief Weight for original images in the blend between original and classified images.
 */
const double ORIGINAL_IMAGE_WEIGHT = 0.6;

/**
 * @brief Threshold used for Intersection over Union (IoU) calculation.
 */
const double IOU_THRESHOLD = 0.5;

/**
 * A constant threshold used to determine if car is inside a parking spot.
 * If more than PERCENTAGE_INSIDE_THRESHOLD*100% of the component is inside the parking spot, it is classified as 'inside'.
 */
const float PERCENTAGE_INSIDE_THRESHOLD = 0.7;

/**
 * @brief Type of image (3-channel 8-bit color image).
 */
const int IMAGE_TYPE = CV_8UC3;

/**
 * @brief Shift used for adjusting the color when combining images.
 */
const int SHIFT = 0;

/**
 * The index of the base sequence.
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
 * @brief Length of the separation line used for console output formatting.
 */
const int SEPARATION_LINE_LENGTH = 36;

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
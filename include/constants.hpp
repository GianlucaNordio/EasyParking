#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

#include <opencv2/opencv.hpp>

/**
 * @brief If true, the algorithm will detect shadows and mark them. 
 *        It decreases the speed a bit, so if you do not need this feature, set the parameter to false.
 */
const bool SHADES_DETECTION = true;

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
 * @brief Color white used for one channel images.
 */
const cv::Scalar WHITE_ONE_CHANNEL = cv::Scalar(255);

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
 * @brief Color light blue used for the unoccupied parking spots in the minimap.
 */
const cv::Scalar LIGHT_BLUE = cv::Scalar(130, 96, 21);

/**
 * @brief Color black used for the borders of the parking spots in the minimap.
 */
const cv::Scalar BLACK = cv::Scalar(0, 0, 0);

/**
 * @brief Size of the rectangle used for drawing bounding boxes on the minimap.
 */
const cv::Size SIZE_RECT_MINIMAP = cv::Size(60,20);

/**
 * @brief Number of margin pixels blac added at the template.
 */
const double TEMPLATE_BLACK_MARGIN = 4;

/**
 * @brief Threshold used to resolve the overlap of RotatedRect.
 */
const double RESOLVE_OVERLAP_THRESHOLD = 0.1;

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
 * @brief Factor used to scale the RotatedRect.
 */
const double SCALE_FACTOR = 1.5;

/**
 * @brief A constant threshold used to determine if car is inside a parking spot.
 *        If more than PERCENTAGE_INSIDE_THRESHOLD*100% of the component is inside the parking spot, it is classified as 'inside'.
 */
const double PERCENTAGE_INSIDE_THRESHOLD = 0.7;

/**
 * @brief Offset on the direction along which the parking spot is split, when performing additional filtering.
 */
const double SPLIT_DIRECTION_OFFSET = 35;

/**
 * @brief A constante used to transform value into percentage.
 */
const double PERCENTAGE = 100;

/**
 * @brief Fraction of segment length to travel from the leftmost endpoint to place 
 *        the starting point of the perpendicular line of a segment with positive slope.
 */
const double POSITIVE_SLOPE_SCALE = 0.6;

/**
 * @brief Extend segments by this factor when another .
 */
const double EXTENSION_SCALE = 0.4;

/**
 * @brief Fraction of segment length to travel from the leftmost endpoint to place 
 *        the starting point of the perpendicular line of a segment with negative slope.
 */
const double NEGATIVE_SLOPE_SCALE = 0.25;

/**
 * @brief Scale factor used to compute the length for the perpendicular segment.
 */
const double SEARCH_LENGTH_SCALE = 2.5;

/**
 * @brief The minimum allowed distance between two segments to build a rotated rect.
 */
const double LOWER_BOUND_DISTANCE = 22.5;

/**
 * @brief Shift applied to the center of the rectangle to adjust the position of the parking spot.
 *        This is done only to achieve better result in the evaluation step. (See the report)
 */
const double CENTER_SHIFT = 15;

/**
 * @brief The max length of the perpendicular segment used to search lines of the same slope.
 */
const double MAX_SEARCH_LENGTH = 200.0;

/**
 * @brief The minimum area of a rectangle to be considered valid.
 */
const double MIN_AREA = 1;

/**
 * @brief Template height value.
 */
const double TEMPLATE_HEIGHT = 10;

/**
 * @brief Constant used to tell the model to not update its model of the background.
 *        This constant is used by the backgroundSubtraction when performing the prediction 
 *        on an image containing things that are not the background.
 */
const int BACKGROUND_NOT_UPDATED = 0;

/**
 * @brief Type of image (3-channel 8-bit color image).
 */
const int IMAGE_TYPE_3_CHANNELS = CV_8UC3;

/**
 * @brief Type of image (1-channel 8-bit color image).
 */
const int IMAGE_TYPE_1_CHANNEL = CV_8U;

/**
 * @brief Offset used for homography transformation.
 */
const int OFFSET_HOMOGRAPHY = -25;

/**
 * @brief Shift used for adjusting the color when combining images.
 */
const int SHIFT = 0;

/**
 * @brief Rectangular structuring element for morphologyEx.
 */
const int MORPH_RECT = cv::MORPH_CROSS;

/**
 * @brief Size of structuring element for closing.
 */
const int MORPH_SIZE = 1;

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
 * @brief 8-connectivity constant for connectedComponentsWithStats.
 */
const int CONNECTIVITY_8 = 8;

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
 * @brief The maximum allowable difference in rotation angles for the rectangles to be considered aligned.
 */
const double ANGLE_TOLERANCE = 16;

/**
 * @brief Value used for the black area in the mask in template matching.
 */
const int MASK_LINE_VALUE_LOW = 10;

/**
 * @brief Value for the history length parameter of the background subtractor.
 */
const int HISTORY_DEFAULT_VALUE = 500;

/**
 * @brief Threshold on the squared Mahalanobis distance between the pixel and the model to decide whether a pixel 
 *        is well described by the background model. This parameter does not affect the background update.
 */
const int VAR_THRESHOLD = 50;

/**
 * @brief High threshold used for the shadow detection.
 */
const int SHADOW_HIGH_THRESHOLD = 255;

/**
 * @brief Low threshold used for the shadow detection.
 */
const int SHADOW_LOW_THRESHOLD = 128;

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
 * @brief Threshold for the minumum size of the connected component to be kept.
 */
const int PIXEL_SIZE_THRESHOLD = 700;  

/**
 * @brief Height of the minimap.
 */
const int MAP_HEIGHT = 250;

/**
 * @brief Width of the minimap.
 */
const int MAP_WIDTH = 450;

/**
 * @brief X coordinate of Point1 used for create a black mask.
 */
const int BLACK_MASK_x1 = 850;

/**
 * @brief Y coordinate of Point1 used for create a black mask.
 */
const int BLACK_MASK_y1 = 0;

/**
 * @brief X coordinate of Point2 used for create a black mask.
 */
const int BLACK_MASK_x2 = 1280;

/**
 * @brief Y coordinate of Point2 used for create a black mask.
 */
const int BLACK_MASK_y2 = 230;

/**
 * @brief Color used for the black mask.
 */
const int BLACK_MASK_COLOR = 0;

/**
 * @brief X coordinate of Point1 used to identify top-right corner.
 */
const int TOP_RIGHT_CORNER_X1 = 850;

/**
 * @brief Y coordinate of Point1 used to identify top-right corner.
 */
const int TOP_RIGHT_CORNER_Y1 = 0;

/**
 * @brief Y coordinate of Point2 used to identify top-right corner.
 */
const int TOP_RIGHT_CORNER_Y2 = 300;

/**
 * @brief Y coordinate of top-right corner.
 */
const int TOP_RIGHT_CORNER_Y = 0;

#endif // CONSTANTS_HPP
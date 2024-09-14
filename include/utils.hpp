#ifndef UTILS_HPP
#define UTILS_HPP

#include <opencv2/highgui.hpp>

#include "parkingSpot.hpp"
#include "parser.hpp"

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
 * The index of the base sequence.
 */
const int BASE_SEQUENCE_INDEX = 0;

/**
 * Creates a single image by concatenating a sequence of input images in a grid format.
 * The number of images per row is specified by the `imagesPerLine` parameter.
 * The function checks that all images have the same size and adjusts the number of images per row if necessary.
 *
 * @param images         A vector of cv::Mat objects representing the input images to be concatenated.
 * @param imagesPerLine The number of images to place per row in the output image.
 * @return              A cv::Mat object containing the concatenated image.
 * @throws std::invalid_argument if no images are provided or if images have different sizes.
 */
cv::Mat produceSingleImage(const std::vector<cv::Mat>& images, int imagesPerLine);

/**
 * Loads frames from a base sequence dataset directory into a vector of cv::Mat objects.
 * The frames are loaded from a predefined folder path that includes the base sequence index.
 *
 * @param datasetPath The path to the dataset directory.
 * @param images      A reference to a vector of cv::Mat objects where the loaded frames will be stored.
 */
void loadBaseSequenceFrames(const std::string& datasetPath, std::vector<cv::Mat> &images);

/**
 * Loads frames from multiple sequences of a dataset into a vector of vectors of cv::Mat objects.
 * Frames are loaded from directories corresponding to each sequence.
 *
 * @param datasetPath The path to the dataset directory.
 * @param numSequences The number of sequences to load.
 * @param images      A reference to a vector of vectors of cv::Mat objects where the loaded frames will be stored.
 */
void loadSequencesFrames(const std::string& datasetPath, int numSequences, std::vector<std::vector<cv::Mat>> &images);

/**
 * Loads images from a specified directory into a vector of cv::Mat objects.
 * Only regular files are processed, and empty or unreadable images are reported.
 *
 * @param path   The path to the directory containing images.
 * @param images A reference to a vector of cv::Mat objects where the loaded images will be stored.
 * @throws std::invalid_argument if the provided path is not a directory.
 * @throws std::runtime_error if an image file is empty or cannot be read.
 */
void loadImages(const std::string path, std::vector<cv::Mat> &images);

/**
 * Loads segmentation masks for multiple sequences of a dataset into a vector of vectors of cv::Mat objects.
 * Each mask image is converted to grayscale.
 *
 * @param datasetPath The path to the dataset directory.
 * @param numSequences The number of sequences to load.
 * @param segMasks    A reference to a vector of vectors of cv::Mat objects where the loaded masks will be stored.
 */
void loadSequencesSegMasks(const std::string& datasetPath, int numSequences, std::vector<std::vector<cv::Mat>> &segMasks);

/**
 * Loads ground truth data for the base sequence from a specified dataset path.
 * The ground truth data is read from an XML file located in the base sequence directory.
 *
 * @param datasetPath The path to the dataset directory.
 * @param groundTruth A reference to a vector of ParkingSpot objects where the loaded ground truth data will be stored.
 */
void loadBaseSequenceGroundTruth(const std::string& datasetPath, std::vector<ParkingSpot> &groundTruth);

/**
 * Loads ground truth data for multiple sequences from a specified dataset path.
 * Each sequence's ground truth data is read from an XML file located in its corresponding directory.
 *
 * @param datasetPath The path to the dataset directory.
 * @param numSequences The number of sequences to load.
 * @param groundTruth A reference to a vector of vectors of ParkingSpot objects where the loaded ground truth data will be stored.
 */
void loadSequencesGroundTruth(const std::string& datasetPath, int numSequences, std::vector<std::vector<ParkingSpot>> &groundTruth);

/**
 * Loads ground truth data from an XML file into a vector of ParkingSpot objects.
 *
 * @param path        The path to the XML file containing the ground truth data.
 * @param groundTruth A reference to a vector of ParkingSpot objects where the loaded ground truth data will be stored.
 */
void loadGroundTruth(const std::string path, std::vector<ParkingSpot> &groundTruth);

/**
 * Retrieves the name of the first file found in a specified folder.
 *
 * @param folderPath The path to the folder.
 * @return           The name of the first file found in the folder, or an empty string if no files are found.
 */
std::string getFirstFileInFolder(const std::string& folderPath);

/**
 * Converts a vector of grayscale images to a vector of BGR images.
 * The grayscale values are mapped to specific BGR colors.
 *
 * @param greyImages A vector of vectors of grayscale images to be converted.
 * @param BGRImages  A reference to a vector of vectors of BGR images where the converted images will be stored.
 */
void convertGreyMasksToBGR(const std::vector<std::vector<cv::Mat>> &greyImages, std::vector<std::vector<cv::Mat>> &BGRImages);

/**
 * Converts a vector of grayscale images to BGR images.
 * The grayscale values are mapped to specific BGR colors.
 *
 * @param greyImage A vector of grayscale images to be converted.
 * @param BGRImage  A reference to a vector of BGR images where the converted images will be stored.
 */
void convertGreyMaskToBGR(const std::vector<cv::Mat> &greyImage, std::vector<cv::Mat> &BGRImage);

/**
 * Prints performance metrics including mean Average Precision (mAP) and Intersection over Union (IoU) for each frame.
 * The metrics are formatted in a table with fixed precision.
 *
 * @param mAPs A vector of mean Average Precision values for each frame.
 * @param IoUs A vector of Intersection over Union values for each frame.
 */
void printPerformanceMetrics(const std::vector<double>& mAPs, const std::vector<double>& IoUs);

/**
 * Draws bounding boxes for parking spots on the given sequence of images.
 *
 * For each image in the base sequence, this function creates a copy of the image and draws 
 * bounding boxes around each parking spot using the rectangle coordinates from the `ParkingSpot` objects.
 * The bounding boxes are drawn in red with a thickness of 2 pixels.
 * The modified images with bounding boxes are then stored in the `baseSequenceBBoxes` vector.
 *
 * @param parkingSpot A vector of `ParkingSpot` objects representing the parking spots to be drawn.
 * @param baseSequence A vector of `cv::Mat` objects where each `cv::Mat` represents an image in the base sequence.
 * @param baseSequenceBBoxes A vector of `cv::Mat` objects where each `cv::Mat` will store the image with drawn bounding boxes.
 */
void printParkingSpot(const std::vector<ParkingSpot>& parkingSpot, const std::vector<cv::Mat>& baseSequence, std::vector<cv::Mat>& baseSequenceBBoxes);

#endif // UTILS_HPP
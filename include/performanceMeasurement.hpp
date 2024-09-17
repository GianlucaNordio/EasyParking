// Giovanni Cinel

#ifndef PERFORMANCE_MEASUREMENT_H
#define PERFORMANCE_MEASUREMENT_H

#include <iostream>
#include <vector>

#include "parkingSpot.hpp"
#include "classification.hpp"
#include "utils.hpp"
#include "constants.hpp"

/**
 * Computes performance metrics for a set of parking spot detection results.
 *
 * This function evaluates the performance of parking spot detection algorithms on a base sequence and
 * a dataset of sequences. It calculates metrics such as Mean Average Precision (mAP) and Mean Intersection
 * over Union (mIoU) for each frame in the base sequence and the dataset sequences. The function also computes
 * average mAP and IoU values for both the base sequence and each sequence on the dataset.
 * 
 * @param DATASET_PATH The base path to the dataset containing the ground truth and mask images.
 * @param NUMBER_SEQUENCES The number of sequences in the dataset.
 * @param baseSequenceParkingSpot A vector of vector of `ParkingSpot` objects representing the detected parking spots in the base sequence used for evaluation.
 * @param datasetParkingSpot A vector of vector of vector of `ParkingSpot` objects representing the detected parking spots in the dataset used for evaluation.
 * @param baseSequence A vector of `cv::Mat` representing the images in the base sequence.
 * @param dataset A vector of vectors of `cv::Mat` where each inner vector represents the images for a sequence in the dataset.
 * @param classifiedDatasetMasks A vector of vectors of `cv::Mat` where each inner vector contains the results of classification task for a sequence in the dataset.
 * @param classifiedBaseSequenceMasks A vector of `cv::Mat` containing the result of classification task for the base sequence.
 * @param baseSequenceMAP A vector of doubles where each element represents the Mean Average Precision for a frame in the base sequence.
 * @param baseSequenceIoU A vector of doubles where each element represents the Mean Intersection over Union for a frame in the base sequence.
 * @param averageBaseSequenceMAP A double representing the average Mean Average Precision across the base sequence.
 * @param averageBaseSequenceIoU A double representing the average Mean Intersection over Union across the base sequence.
 * @param datasetMAP A vector of vectors of doubles where each inner vector contains the Mean Average Precision values for a sequence in the dataset.
 * @param datasetIoU A vector of vectors of doubles where each inner vector contains the Mean Intersection over Union values for a sequence in the dataset.
 * @param averageDatasetMAP A vector of doubles where each element represents the average Mean Average Precision for a sequence in the dataset.
 * @param averageDatasetIoU A vector of doubles where each element represents the average Mean Intersection over Union for a sequence in the dataset.
 */
void performanceMeasurement(const std::string DATASET_PATH, const int NUMBER_SEQUENCES, const std::vector<std::vector<ParkingSpot>>& baseSequenceParkingSpot, const std::vector<std::vector<std::vector<ParkingSpot>>>& datasetParkingSpot,
        const std::vector<cv::Mat>& baseSequence, const std::vector<std::vector<cv::Mat>>& dataset, const std::vector<std::vector<cv::Mat>>& classifiedDatasetMasks, const std::vector<cv::Mat>& classifiedBaseSequenceMasks,
        std::vector<double>& baseSequenceMAP, std::vector<double>& baseSequenceIoU, double& averageBaseSequenceMAP, double& averageBaseSequenceIoU, std::vector<std::vector<double>>& datasetMAP,
        std::vector<std::vector<double>>& datasetIoU, std::vector<double>& averageDatasetMAP, std::vector<double>& averageDatasetIoU);

/**
 * Calculates the mean Average Precision (mAP) for parking spot predictions.
 *
 * The function divides the predictions and ground truths into two classes: car inside parking spot and car outside parking spot.
 * It then calculates the Precision-Recall curve for each class, computes the Average Precision (AP) for each class, 
 * and finally computes the mean Average Precision (mAP) by averaging the APs of both classes.
 *
 * @param predictions   A vector of ParkingSpot objects representing the predicted parking spots.
 * @param groundTruths  A vector of ParkingSpot objects representing the ground truth parking spots.
 * @return              The mean Average Precision (mAP) for the parking spot predictions.
 */
double calculateMeanAveragePrecision(const std::vector<ParkingSpot>& predictions, const std::vector<ParkingSpot>& groundTruths);

/**
 * Calculates the mean Intersection over Union (mIoU) for the given masks.
 *
 * The mean IoU is computed for three different classes: background, car inside parking spot,
 * and car outside parking spot. The IoU for each class is calculated using the `classIoU` function
 * and the mean of these values is returned.
 *
 * @param foundMask        The mask found by the model, as a single-channel cv::Mat.
 * @param groundTruthMask  The ground truth mask, as a single-channel cv::Mat.
 * @return                 The mean Intersection over Union (mIoU) between the found mask
 *                         and the ground truth mask.
 * @throws std::invalid_argument if either of the masks is empty.
 */
double calculateMeanIntersectionOverUnion(const cv::Mat &foundMask, const cv::Mat &groundTruthMask);

/**
 * Calculates the Precision-Recall curve points for parking spot predictions against the ground truth.
 *
 * The function sorts the predicted parking spots by confidence in descending order and computes the
 * precision and recall values at each prediction step. It returns a vector of (recall, precision) pairs
 * that form the Precision-Recall curve.
 *
 * @param predictions    A vector of ParkingSpot objects representing the predicted parking spots.
 * @param groundTruths  A vector of ParkingSpot objects representing the ground truth parking spots.
 * @return              A vector of pairs, where each pair contains the recall and precision values.
 */
std::vector<std::pair<double, double>> calculatePrecisionRecallCurve(const std::vector<ParkingSpot>& predictions, const std::vector<ParkingSpot>& groundTruths);

/**
 * Computes the area of intersection between two rotated rectangles.
 * 
 * This method calculates the overlapping area of two rotated rectangles
 * by obtaining their vertices, computing the convex hull of the intersection,
 * and then calculating the intersection area.
 * 
 * @param rect1 The first rotated rectangle.
 * @param rect2 The second rotated rectangle.
 * @return The area of the intersection between the two rotated rectangles. 
 *         If there is no intersection, the returned area will be 0.
 * 
 * @note This method uses OpenCV's `intersectConvexConvex` function to calculate 
 *       the area of intersection.
 */
double computeIntersectionArea(const cv::RotatedRect& rect1, const cv::RotatedRect& rect2);

/**
 * Calculates the Intersection over Union (IoU) between two parking spots.
 *
 * The IoU is computed as the area of the intersection between two parking spots divided by the area
 * of their union. It checks for intersection between the two rotated rectangles representing parking spots
 * and calculates the IoU accordingly.
 *
 * @param parkingSpot1  The first ParkingSpot object.
 * @param parkingSpot2  The second ParkingSpot object.
 * @return              The IoU value between the two parking spots. Returns 0.0 if they do not intersect.
 */
double calculateIoU(const ParkingSpot& rect1, const ParkingSpot& rect2);

/**
 * Calculates the Average Precision (AP) given a set of Precision-Recall points.
 *
 * The AP is computed as the average of the interpolated precision values at specified recall levels.
 * It interpolates the precision values over a range of recall levels from 0.0 to 1.0.
 *
 * @param precisionRecallPoints  A vector of pairs, where each pair contains recall and precision values.
 * @return                       The Average Precision (AP) value.
 */
double calculateAveragePrecision(const std::vector<std::pair<double, double>>& precisionRecallPoints);

/**
 * Computes the Intersection over Union (IoU) for a specific class between the found mask and the ground truth mask.
 *
 * The IoU is calculated as the ratio of the area of intersection to the area of union for a given class.
 * It first generates binary masks for the specified class in both the found and ground truth masks,
 * and then uses these masks to compute the intersection and union.
 *
 * @param foundMask        The mask found by the model, as a single-channel cv::Mat.
 * @param groundTruthMask  The ground truth mask, as a single-channel cv::Mat.
 * @param id               The class label ID for which IoU is to be calculated.
 * @return                 The Intersection over Union (IoU) value for the specified class label.
 *                         Returns 1 if there is no area for the union.
 */
double classIoU(const cv::Mat &foundMask, const cv::Mat &groundTruthMask, labelId id);

#endif
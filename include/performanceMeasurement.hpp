#ifndef PERFORMANCE_MEASUREMENT_H
#define PERFORMANCE_MEASUREMENT_H

#include <iostream>
#include <vector>

#include "parkingSpot.hpp"
#include "classification.hpp"

const float IOU_THRESHOLD = 0.5;

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
double calculateMeanAveragePrecision(const std::vector<ParkingSpot>& groundTruths, const std::vector<ParkingSpot>& detections);

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
 * @param groundTruths  A vector of ParkingSpot objects representing the ground truth parking spots.
 * @param detections    A vector of ParkingSpot objects representing the predicted parking spots.
 * @return              A vector of pairs, where each pair contains the recall and precision values.
 */
std::vector<std::pair<double, double>> calculatePrecisionRecallCurve(const std::vector<ParkingSpot>& groundTruths, const std::vector<ParkingSpot>& detections);

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
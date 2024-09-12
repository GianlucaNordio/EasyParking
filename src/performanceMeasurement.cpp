#include "performanceMeasurement.hpp"

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
double calculateMeanAveragePrecision(const std::vector<ParkingSpot>& predictions, const std::vector<ParkingSpot>& groundTruths) {
    
    // Divide the predictions in two classes: car inside parking spot and car outside parking spot
    std::vector<ParkingSpot> predictionsParkingSpotWithCar;
    std::vector<ParkingSpot> predictionsParkingSpotWithoutCar;

    for (const auto& prediction : predictions) {
        if (prediction.occupied)
            predictionsParkingSpotWithCar.push_back(prediction);
        else
            predictionsParkingSpotWithoutCar.push_back(prediction);
    }

    // Divide the ground truths in two classes: car inside parking spot and car outside parking spot
    std::vector<ParkingSpot> groundTruthsParkingSpotWithCar;
    std::vector<ParkingSpot> groundTruthsParkingSpotWithoutCar;

    for (const auto& groundTruth : groundTruths) {
        if (groundTruth.occupied)
            groundTruthsParkingSpotWithCar.push_back(groundTruth);
        else
            groundTruthsParkingSpotWithoutCar.push_back(groundTruth);
    }

    double APParkingSpotWithCar = 1;
    double APParingSpotWithoutCar = 1;

    // Check if there are no predictions for ParkingSpot with car
    if (!predictionsParkingSpotWithCar.empty()){
        // Calculate the Precision-Recall curve for ParkingSpot with car
        std::vector<std::pair<double, double>> precisionRecallPointsWithCar = calculatePrecisionRecallCurve(groundTruthsParkingSpotWithCar, predictionsParkingSpotWithCar);
        // Calculate the Average Precision (AP) for ParkingSpot with car
        APParkingSpotWithCar = calculateAveragePrecision(precisionRecallPointsWithCar);
    }

    // Check if there are no predictions for ParkingSpot without car
    if (!predictionsParkingSpotWithoutCar.empty()){
        // Calculate the Precision-Recall curve for ParkingSpot without car
        std::vector<std::pair<double, double>> precisionRecallPointsWithoutCar = calculatePrecisionRecallCurve(groundTruthsParkingSpotWithoutCar, predictionsParkingSpotWithoutCar);
        // Calculate the Average Precision (AP) for ParkingSpot without car
        APParingSpotWithoutCar = calculateAveragePrecision(precisionRecallPointsWithoutCar);
    }
    

    // Calculate the mean Average Precision (mAP) for the two classes
    double mAP = (APParkingSpotWithCar + APParingSpotWithoutCar) / 2;

    return mAP;
}

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
std::vector<std::pair<double, double>> calculatePrecisionRecallCurve(const std::vector<ParkingSpot>& groundTruths, const std::vector<ParkingSpot>& detections){

    std::vector<std::pair<double, double>> precisionRecallPoints;
    int truePositives = 0;
    int falsePositives = 0;

    std::vector<bool> detected(groundTruths.size(), false);

    // Sort the predictions by confidence in descending order to ensure highest confidence detections are considered first
    std::vector<ParkingSpot> sortedPredictions = detections;
    std::sort(sortedPredictions.begin(), sortedPredictions.end(), 
              [](const ParkingSpot& a, const ParkingSpot& b) {
                  return a.confidence > b.confidence; 
              });

    // Calculate Precision-Recall points for each prediction
    for (const auto& prediction : sortedPredictions) {
        double maxIoU = 0.0;
        int bestMatchIndex = -1;

        for (size_t i = 0; i < groundTruths.size(); ++i) {
            if (detected[i]) continue; 

            double iou = calculateIoU(prediction, groundTruths[i]);
            if (iou > maxIoU) {
                maxIoU = iou;
                bestMatchIndex = i;
            }
        }

        if (maxIoU >= IOU_THRESHOLD) {
            truePositives++;
            detected[bestMatchIndex] = true;
        } else {
            falsePositives++;
        }

        double precision = truePositives / static_cast<double>(truePositives + falsePositives);
        double recall = truePositives / static_cast<double>(groundTruths.size());
        
        precisionRecallPoints.emplace_back(recall, precision);
    }

    return precisionRecallPoints;
}

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
double calculateIoU(const ParkingSpot& parkingSpot1, const ParkingSpot& parkingSpot2) {

    // Check if the two rectangles intersect
    std::vector<cv::Point2f> intersectionPoints;
    int intersectionType = cv::rotatedRectangleIntersection(parkingSpot1.rect, parkingSpot2.rect, intersectionPoints);
    if (intersectionPoints.empty() || intersectionType == cv::INTERSECT_NONE) {
        return 0.0; 
    }    
    
    // Calculate the Intersection over Union (IoU)
    double intersectionArea = cv::contourArea(intersectionPoints);
    double areaRect1 = parkingSpot1.rect.size.area();
    double areaRect2 = parkingSpot2.rect.size.area();
    double iou = intersectionArea / (areaRect1 + areaRect2 - intersectionArea);

    return iou;
}

/**
 * Calculates the Average Precision (AP) given a set of Precision-Recall points.
 *
 * The AP is computed as the average of the interpolated precision values at specified recall levels.
 * It interpolates the precision values over a range of recall levels from 0.0 to 1.0.
 *
 * @param precisionRecallPoints  A vector of pairs, where each pair contains recall and precision values.
 * @return                       The Average Precision (AP) value.
 */
double calculateAveragePrecision(const std::vector<std::pair<double, double>>& precisionRecallPoints) {
    std::vector<double> recallLevels = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    std::vector<double> interpolatedPrecisions(recallLevels.size(), 0.0);

    // Interpolate the precision values at the given recall levels
    for (size_t i = 0; i < recallLevels.size(); i++) {
        double recallLevel = recallLevels[i];
        double maxPrecision = 0.0;
        for (const auto& point : precisionRecallPoints) {
            double recall = point.first;
            double precision = point.second;
            if (recall >= recallLevel) {
                maxPrecision = std::max(maxPrecision, precision);
            }
        }
        interpolatedPrecisions[i] = maxPrecision;
    }

    // Calculate the Average Precision (AP) as the mean of the interpolated precision values
    double AP = 0.0;
    for (double precision : interpolatedPrecisions) {
        AP += precision;
    }
    AP /= recallLevels.size();

    return AP;
}

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
double calculateMeanIntersectionOverUnion(const cv::Mat &foundMask, const cv::Mat &groundTruthMask){
    
    // Check if the masks are empty
    if (foundMask.empty() || groundTruthMask.empty())
    {
        const std::string INVALID_EMPTY_MAT = "Masks cannot be empty.";
        throw std::invalid_argument(INVALID_EMPTY_MAT);
    }

    // Calculate the IoU for each class
    double background = classIoU(foundMask, groundTruthMask, labelId::background);
    double carInsideParkingSpot = classIoU(foundMask, groundTruthMask, labelId::carInsideParkingSpot);
    double carOutsideParkingSpot = classIoU(foundMask, groundTruthMask, labelId::carOutsideParkingSpot);
    
    // Calculate the mean IoU
    double mIoU = (background + carInsideParkingSpot + carOutsideParkingSpot) / 3;

    return mIoU;
}

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

double classIoU(const cv::Mat &foundMask, const cv::Mat &groundTruthMask, labelId id){
    
    // Check if the masks are single-channel
    CV_Assert(foundMask.channels() == 1);
    CV_Assert(groundTruthMask.channels() == 1);

    // Generate binary masks for the specified class
    cv::Mat foundClassMask, groundTruthClassMask;
    cv::inRange(foundMask, cv::Scalar(id), cv::Scalar(id), foundClassMask);
    cv::inRange(groundTruthMask, cv::Scalar(id), cv::Scalar(id), groundTruthClassMask);

    // Compute the union and intersection of the masks
    cv::Mat UnionClassMask, IntersectionClassMask;
    cv::bitwise_or(foundClassMask, groundTruthClassMask, UnionClassMask);
    cv::bitwise_and(foundClassMask, groundTruthClassMask, IntersectionClassMask);

    double unionArea = static_cast<double>(cv::countNonZero(UnionClassMask));
    double intersectionArea = static_cast<double>(cv::countNonZero(IntersectionClassMask));
    
    // Return 1 if there is no area for the union
    if (unionArea == 0)
        return 1;
    
    // Calculate the IoU for the specified class
    double IoU = intersectionArea / unionArea;

    return IoU;
}
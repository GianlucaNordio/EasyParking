#include "performanceMeasurement.hpp"

double calculateMeanAveragePrecision(const std::vector<ParkingSpot>& predictions, const std::vector<ParkingSpot>& groundTruths) {
    
    
    std::vector<std::pair<double, double>> precisionRecallPoints = calculatePrecisionRecallCurve(groundTruths, predictions);
    return calculateAveragePrecision(precisionRecallPoints);
}

std::vector<std::pair<double, double>> calculatePrecisionRecallCurve(const std::vector<ParkingSpot>& groundTruths, const std::vector<ParkingSpot>& detections){

    std::vector<std::pair<double, double>> precisionRecallPoints;
    int truePositives = 0;
    int falsePositives = 0;

    std::vector<bool> detected(groundTruths.size(), false);

    std::vector<ParkingSpot> sortedPredictions = detections;
    std::sort(sortedPredictions.begin(), sortedPredictions.end(), 
              [](const ParkingSpot& a, const ParkingSpot& b) {
                  return a.confidence > b.confidence; 
              });

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

double calculateIoU(const ParkingSpot& parkingSpot1, const ParkingSpot& parkingSpot2) {

    
    std::vector<cv::Point2f> intersectionPoints;
    int intersectionType = cv::rotatedRectangleIntersection(parkingSpot1.rect, parkingSpot2.rect, intersectionPoints);

    if (intersectionPoints.empty() || intersectionType == cv::INTERSECT_NONE) {
        return 0.0; 
    }    
    

    double intersectionArea = cv::contourArea(intersectionPoints);
    double areaRect1 = parkingSpot1.rect.size.area();
    double areaRect2 = parkingSpot2.rect.size.area();

    double iou = intersectionArea / (areaRect1 + areaRect2 - intersectionArea);
    return iou;
}


double calculateAveragePrecision(const std::vector<std::pair<double, double>>& precisionRecallPoints) {
    std::vector<double> recallLevels = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    std::vector<double> interpolatedPrecisions(recallLevels.size(), 0.0);

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
    if (foundMask.empty() || groundTruthMask.empty())
    {
        const std::string INVALID_EMPTY_MAT = "Masks cannot be empty.";
        throw std::invalid_argument(INVALID_EMPTY_MAT);
    }

    double background = classIoU(foundMask, groundTruthMask, labelId::background);
    double carInsideParkingSpot = classIoU(foundMask, groundTruthMask, labelId::carInsideParkingSpot);
    double carOutsideParkingSpot = classIoU(foundMask, groundTruthMask, labelId::carOutsideParkingSpot);
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
    CV_Assert(foundMask.channels() == 1);
    CV_Assert(groundTruthMask.channels() == 1);

    cv::Mat foundClassMask, groundTruthClassMask;
    cv::inRange(foundMask, cv::Scalar(id), cv::Scalar(id), foundClassMask);
    cv::inRange(groundTruthMask, cv::Scalar(id), cv::Scalar(id), groundTruthClassMask);
    cv::Mat UnionClassMask, IntersectionClassMask;
    cv::bitwise_or(foundClassMask, groundTruthClassMask, UnionClassMask);
    cv::bitwise_and(foundClassMask, groundTruthClassMask, IntersectionClassMask);


    double unionArea = static_cast<double>(cv::countNonZero(UnionClassMask));
    if (unionArea == 0)
        return 1;
    float intersectionArea = static_cast<double>(cv::countNonZero(IntersectionClassMask));
    return intersectionArea / unionArea;
}
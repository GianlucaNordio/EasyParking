#include "performanceMeasurement.hpp"

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

double calculateAveragePrecision(const std::vector<std::pair<double, double>>& precisionRecallPoints) {
    double AP = 0.0;
    double previousRecall = 0.0;

    for (const auto& point : precisionRecallPoints) {
        double recall = point.first;
        double precision = point.second;
        AP += precision * (recall - previousRecall);
        previousRecall = recall;
    }

    return AP;
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

double calculateMeanAveragePrecision(const std::vector<ParkingSpot>& predictions, 
                    const std::vector<ParkingSpot>& groundTruths) {
    std::vector<std::pair<double, double>> precisionRecallPoints = calculatePrecisionRecallCurve(predictions, groundTruths);
    return calculateAveragePrecision(precisionRecallPoints);
}

double meanIntersectionOverUnion(){
    return 0.0;
}
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
    std::vector<std::pair<double, double>> precisionRecallPoints = calculatePrecisionRecallCurve(groundTruths, predictions);
    return calculateAveragePrecision(precisionRecallPoints);
}

double calculateMeanIntersectionOverUnion(const std::vector<cv::Mat> &foundMask, const std::vector<cv::Mat> &groundTruthMask){
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
double classIoU(const std::vector<cv::Mat> &foundMask, const std::vector<cv::Mat> &groundTruthMask, labelId id){
    
    double classIoU = 0;
    for (int i = 0; i < foundMask.size(); i++)
        classIoU += singleImmageClassIoU(foundMask.at(i), groundTruthMask.at(i), id);

    return classIoU / foundMask.size();
}

double singleImmageClassIoU(const cv::Mat &foundMask, const cv::Mat &groundTruthMask, labelId id){
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
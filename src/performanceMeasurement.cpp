#include "performanceMeasurement.hpp"

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
 * @param parkingSpot A vector of `ParkingSpot` objects representing the detected parking spots for evaluation.
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
void performanceMeasurement(const std::string DATASET_PATH, const int NUMBER_SEQUENCES, const std::vector<ParkingSpot>& parkingSpot, const std::vector<cv::Mat>& baseSequence, 
    const std::vector<std::vector<cv::Mat>>& dataset, const std::vector<std::vector<cv::Mat>>& classifiedDatasetMasks, const std::vector<cv::Mat>& classifiedBaseSequenceMasks,
    std::vector<double>& baseSequenceMAP, std::vector<double>& baseSequenceIoU, double& averageBaseSequenceMAP, double& averageBaseSequenceIoU, 
    std::vector<std::vector<double>>& datasetMAP, std::vector<std::vector<double>>& datasetIoU, std::vector<double>& averageDatasetMAP, std::vector<double>& averageDatasetIoU) {
    
    // Load the ground truth
    cv::Mat baseSequenceMaskGT = cv::Mat::zeros(baseSequence[0].size(), CV_8UC1);
    std::vector<std::vector<ParkingSpot>> baseSequenceParkingSpotGT;
    std::vector<std::vector<std::vector<ParkingSpot>>> datasetParkingSpotGT;
    std::vector<std::vector<cv::Mat>> sequenceMaskGTGray;
    std::vector<std::vector<cv::Mat>> sequenceMaskGTBGR;

    loadBaseSequenceGroundTruth(DATASET_PATH, baseSequenceParkingSpotGT);
    loadSequencesGroundTruth(DATASET_PATH, NUMBER_SEQUENCES, datasetParkingSpotGT);
    loadSequencesSegMasks(DATASET_PATH, NUMBER_SEQUENCES, sequenceMaskGTGray);

    // Compute performance for the base sequence
    
    for(int i = 0; i < baseSequence.size(); i++) {
        baseSequenceMAP.push_back(calculateMeanAveragePrecision(parkingSpot, baseSequenceParkingSpotGT[i]));
        baseSequenceIoU.push_back(calculateMeanIntersectionOverUnion(classifiedBaseSequenceMasks[i], baseSequenceMaskGT));
    }

    averageBaseSequenceMAP = 0;
    averageBaseSequenceIoU = 0;

    for(int i = 0; i < baseSequenceMAP.size(); i++) {
        averageBaseSequenceMAP += baseSequenceMAP[i];
        averageBaseSequenceIoU += baseSequenceIoU[i];
    }

    averageBaseSequenceMAP /= baseSequenceMAP.size();
    averageBaseSequenceIoU /= baseSequenceIoU.size();

    // Compute performance for the dataset


    for(int i = 0; i < NUMBER_SEQUENCES; i++) {
        std::vector<double> sequenceMAP;
        std::vector<double> sequenceIoU;

        for(int j = 0; j < dataset[i].size(); j++) {
            sequenceMAP.push_back(calculateMeanAveragePrecision(parkingSpot, datasetParkingSpotGT[i][j]));
            sequenceIoU.push_back(calculateMeanIntersectionOverUnion(classifiedDatasetMasks[i][j], sequenceMaskGTGray[i][j]));
        }

        datasetMAP.push_back(sequenceMAP);
        datasetIoU.push_back(sequenceIoU);
    }

    for(int i = 0; i < datasetMAP.size(); i++) {
        double averageSequenceMAP = 0;
        double averageSequenceIoU = 0;

        for(int j = 0; j < datasetMAP[i].size(); j++) {
            averageSequenceMAP += datasetMAP[i][j];
            averageSequenceIoU += datasetIoU[i][j];
        }

        averageSequenceMAP /= datasetMAP[i].size();
        averageSequenceIoU /= datasetIoU[i].size();

        averageDatasetMAP.push_back(averageSequenceMAP);
        averageDatasetIoU.push_back(averageSequenceIoU);
    }
}

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
    double APParkingSpotWithoutCar = 1;

    // Check if there are no predictions for ParkingSpot with car
    if (!groundTruthsParkingSpotWithCar.empty()){
        // Calculate the Precision-Recall curve for ParkingSpot with car
        std::vector<std::pair<double, double>> precisionRecallPointsWithCar = calculatePrecisionRecallCurve(predictionsParkingSpotWithCar, groundTruthsParkingSpotWithCar);
        // Calculate the Average Precision (AP) for ParkingSpot with car
        APParkingSpotWithCar = calculateAveragePrecision(precisionRecallPointsWithCar);
    }

    // Check if there are no predictions for ParkingSpot without car
    if (!groundTruthsParkingSpotWithoutCar.empty()){
        // Calculate the Precision-Recall curve for ParkingSpot without car
        std::vector<std::pair<double, double>> precisionRecallPointsWithoutCar = calculatePrecisionRecallCurve(predictionsParkingSpotWithoutCar, groundTruthsParkingSpotWithoutCar);
        // Calculate the Average Precision (AP) for ParkingSpot without car
        APParkingSpotWithoutCar = calculateAveragePrecision(precisionRecallPointsWithoutCar);
    }
    

    // Calculate the mean Average Precision (mAP) for the two classes
    double mAP = (APParkingSpotWithCar + APParkingSpotWithoutCar) / 2;

    return mAP;
}

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
std::vector<std::pair<double, double>> calculatePrecisionRecallCurve(const std::vector<ParkingSpot>& predictions, const std::vector<ParkingSpot>& groundTruths){

    std::vector<std::pair<double, double>> precisionRecallPoints;
    int truePositives = 0;
    int falsePositives = 0;
    int falseNegatives = 0;  // GT not detected

    std::vector<bool> detected(groundTruths.size(), false);

    // Sort the predictions by confidence in descending order to ensure highest confidence detections are considered first
    std::vector<ParkingSpot> sortedPredictions = predictions;
    std::sort(sortedPredictions.begin(), sortedPredictions.end(), 
              [](const ParkingSpot& a, const ParkingSpot& b) {
                  return a.confidence > b.confidence; 
              });

    // Calculate Precision-Recall points for each prediction
    for (const auto& prediction : sortedPredictions) {
        double maxIoU = 0.0;
        int bestMatchIndex = -1;

        // Find the best match for the prediction
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
            detected[bestMatchIndex] = true;  // Best match found, it's a TP
        } else {
            falsePositives++;  // No match found, it's a FP
        }

        // Compute precision and recall
        double precision = truePositives / static_cast<double>(truePositives + falsePositives);
        double recall = truePositives / static_cast<double>(groundTruths.size());
        
        precisionRecallPoints.emplace_back(recall, precision);
    }

    double precision = 0.0;
    double recall = 0.0;
    if((truePositives + falsePositives) != 0)
        precision = truePositives / static_cast<double>(truePositives + falsePositives);

    // Add false negatives (GT not detected)
    if(predictions.size() < groundTruths.size()) {
        // Add false negatives (GT not detected)
        for (size_t i = 0; i < groundTruths.size(); ++i) {
            if (!detected[i]) {
                falseNegatives++;
                recall = truePositives / static_cast<double>(truePositives + falseNegatives);

                precisionRecallPoints.emplace_back(recall, precision);
            }     
        }
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

    double intersectionArea = computeIntersectionArea(parkingSpot1.rect, parkingSpot2.rect);

    double areaRect1 = parkingSpot1.rect.size.area();
    double areaRect2 = parkingSpot2.rect.size.area();

    std::cout << "Intersection Area: " << intersectionArea << " Area Rect 1: " << areaRect1 << " Area Rect 2: " << areaRect2 << std::endl;

    double iou = intersectionArea / (areaRect1 + areaRect2 - intersectionArea);

    return iou;
}

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
double computeIntersectionArea(const cv::RotatedRect& rect1, const cv::RotatedRect& rect2) {
    std::vector<cv::Point2f> points1, points2;
    cv::Point2f vertices1[4], vertices2[4];

    double area1 = rect1.size.area();
    double area2 = rect2.size.area();
    
    rect1.points(vertices1);
    rect2.points(vertices2);
    
    for (int i = 0; i < 4; i++) {
        points1.push_back(vertices1[i]);
        points2.push_back(vertices2[i]);
    }

    std::vector<cv::Point2f> intersection;
    double intersectionArea = cv::intersectConvexConvex(points1, points2, intersection);

    return intersectionArea;
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
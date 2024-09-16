#include "lineUtils.hpp"

/**
 * @brief Preprocesses an image to enhance parking lines for detection.
 *
 * This function applies several image processing techniques to the input image in order to
 * highlight parking lines. It uses a bilateral filter for noise reduction while preserving
 * edges, converts the image to grayscale, and computes the gradients along the x and y axes
 * using the Sobel operator. The gradients are then combined, and adaptive thresholding is
 * applied to create a binary image. The result is further refined using morphological operations
 * (dilation and erosion) to enhance the parking lines.
 *
 * Steps:
 * 1. Apply bilateral filtering to reduce noise while preserving edges.
 * 2. Convert the image to grayscale.
 * 3. Compute x and y gradients using the Sobel operator.
 * 4. Combine the gradients to obtain the gradient magnitude.
 * 5. Apply adaptive thresholding to create a binary image highlighting the lines.
 * 6. Use morphological operations (dilation and erosion) to further enhance the detected lines.
 *
 * @param image The input image (cv::Mat) in which to find parking lines.
 * 
 * @return cv::Mat A binary image (cv::Mat) where the parking lines are enhanced for detection.
 */
cv::Mat preprocessFindParkingLines(const cv::Mat& image) {
    cv::Mat filteredImage;
    cv::bilateralFilter(image, filteredImage, -1, 40, 10);

    cv::Mat grayImage;
    cv::cvtColor(filteredImage, grayImage, cv::COLOR_BGR2GRAY);

    cv::Mat gradientX, gradientY;
    cv::Sobel(grayImage, gradientX, CV_16S, 1, 0);
    cv::Sobel(grayImage, gradientY, CV_16S, 0, 1);

    cv::Mat absGradientX;
    cv::Mat absGradientY;
    cv::convertScaleAbs(gradientX, absGradientX);
    cv::convertScaleAbs(gradientY, absGradientY);

    cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3,3)); 

    cv::Mat gradientMagnitude;
    cv::Mat gradientMagnitudeProcessed;
    cv::addWeighted(absGradientX, 0.5, absGradientY, 0.5, 0, gradientMagnitude);
    cv::adaptiveThreshold(gradientMagnitude, gradientMagnitudeProcessed, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C , cv::THRESH_BINARY, 45, -40);
    cv::dilate(gradientMagnitudeProcessed, gradientMagnitudeProcessed, structuringElement, cv::Point(-1,-1), 1);
    cv::erode(gradientMagnitudeProcessed, gradientMagnitudeProcessed, structuringElement, cv::Point(-1,-1), 1);
    
    return gradientMagnitudeProcessed;
}

/**
 * @brief Preprocesses an image to highlight white lines by applying a series of filters and morphological operations.
 * 
 * This function performs several image processing steps to enhance white lines in the input image. The process includes:
 * - Bilateral filtering to reduce noise while preserving edges.
 * - Conversion to grayscale for further thresholding.
 * - Adaptive thresholding to create a binary image based on local means.
 * - Sobel operator to compute the gradient in both X and Y directions.
 * - Addition of the gradients to obtain the magnitude of edges.
 * - Morphological operations (dilation followed by erosion) to enhance the structure of detected lines.
 * 
 * @param image The input image in BGR format.
 * @return A binary image where white lines are enhanced and emphasized, ready for further processing.
 */
cv::Mat preprocessFindWhiteLines(const cv::Mat& image) {
    cv::Mat filteredImage;
    cv::bilateralFilter(image, filteredImage, -1, 40, 10);

    cv::Mat grayImage;
    cv::cvtColor(filteredImage, grayImage, cv::COLOR_BGR2GRAY);

    cv::Mat adptImage;
    cv::adaptiveThreshold(grayImage, adptImage, 255, cv::ADAPTIVE_THRESH_MEAN_C , cv::THRESH_BINARY, 9, -20);

    cv::Mat gradientX, gradientY;
    cv::Sobel(adptImage, gradientX, CV_8U, 1, 0);
    cv::Sobel(adptImage, gradientY, CV_8U, 0, 1);

    cv::Mat magnitude = gradientX + gradientY;

    cv::Mat structuringElement = cv::getStructuringElement( 
                        cv::MORPH_CROSS, cv::Size(3,3)); 

    cv::Mat morphImage;
    cv::dilate(magnitude, morphImage, structuringElement, cv::Point(-1,-1), 4);
    cv::erode(morphImage, morphImage, structuringElement, cv::Point(-1,-1), 3);

    return morphImage;
}

/**
 * @brief Filters out line segments that are too close to each other based on a specified distance threshold.
 * 
 * This function iterates through a list of 2D line segments (represented by cv::Vec4f, with each vector containing 
 * the coordinates of the two endpoints of a line: (x1, y1, x2, y2)). It compares each segment with the others 
 * and discards those that are within a specified distance threshold. The remaining segments are returned in a new vector.
 * 
 * The function avoids rechecking discarded segments for efficiency by maintaining a boolean vector. 
 * 
 * @param segments A vector of line segments, where each segment is represented by a cv::Vec4f (x1, y1, x2, y2).
 * @param distanceThreshold The threshold distance used to determine whether two segments are considered too close.
 * @return A vector of filtered line segments that are not within the specified distance of each other.
 */
std::vector<cv::Vec4f> filterCloseSegments(const std::vector<cv::Vec4f>& segments, double distanceThreshold) {
    std::vector<cv::Vec4f> filteredSegments;
    std::vector<bool> discarded(segments.size(), false);

    for (size_t i = 0; i < segments.size(); ++i) {
        if (discarded[i]) continue;  // Skip already discarded segments

        const cv::Vec4f& currentSegment = segments[i];
        filteredSegments.push_back(currentSegment);  // Add current segment to result

        // Compare with remaining segments and discard close ones
        for (size_t j = i + 1; j < segments.size(); ++j) {
            if (!discarded[j]) {
                double distance = computeDistanceBetweenSegments(currentSegment, segments[j]);
                if (distance < distanceThreshold) {
                    discarded[j] = true;  // Mark this segment as discarded
                }
            }
        }
    }
    return filteredSegments;
}

/**
 * @brief Computes the Euclidean distance between the midpoints of two line segments.
 * 
 * This function calculates the distance between the midpoints of two given 2D line segments. 
 * The midpoints of the segments are computed using the `computeMidpoint` function, and the 
 * Euclidean distance between these midpoints is calculated using `cv::norm`.
 * 
 * @param segment1 The first line segment, represented as a cv::Vec4f containing the endpoints (x1, y1, x2, y2).
 * @param segment2 The second line segment, represented as a cv::Vec4f containing the endpoints (x1, y1, x2, y2).
 * @return The Euclidean distance between the midpoints of the two segments.
 */
double computeDistanceBetweenSegments(const cv::Vec4f& segment1, const cv::Vec4f& segment2) {
    cv::Point2f midPoint1 = computeMidpoint(segment1);
    cv::Point2f midPoint2 = computeMidpoint(segment2);
    return cv::norm(midPoint1 - midPoint2);  // Euclidean distance between midpoints
}

/**
 * @brief Computes the midpoint of a 2D line segment.
 * 
 * This function calculates the midpoint of a line segment defined by two endpoints (x1, y1) and (x2, y2). 
 * The midpoint is computed as the average of the x and y coordinates of the endpoints.
 * 
 * @param segment A line segment represented as a cv::Vec4f, where the vector contains the endpoints (x1, y1, x2, y2).
 * @return A cv::Point2f representing the midpoint of the segment.
 */
cv::Point2f computeMidpoint(const cv::Vec4f& segment) {
    return cv::Point2f((segment[0] + segment[2]) / 2.0f, (segment[1] + segment[3]) / 2.0f);
}

/**
 * @brief Filters out line segments that are near the top-right corner of the image.
 * 
 * This function removes line segments whose endpoints or midpoints fall near the top-right corner 
 * of the image. A convex hull is defined around the top-right corner, and the function checks 
 * whether each segment is inside this region. Segments that are entirely outside the convex hull 
 * are added to the filtered list.
 * 
 * The hull is defined using three points: the top-right corner of the image, and two additional points 
 * to form a triangular region. The function uses `cv::pointPolygonTest` to check whether the 
 * segment's endpoints and midpoint lie inside the convex hull.
 * 
 * @param segments A vector of line segments, where each segment is represented as cv::Vec4f (x1, y1, x2, y2).
 * @param imageSize The size of the image (width and height), used to determine the top-right corner.
 * @return A vector of filtered segments that are not near the top-right corner of the image.
 */
std::vector<cv::Vec4f> filterSegmentsNearTopRight(const std::vector<cv::Vec4f>& segments, const cv::Size& imageSize) {
    
    // Define the top-right corner of the image
    cv::Point2f topRightCorner(imageSize.width - 1, TOP_RIGHT_CORNER_Y);
    cv::Point2f start(TOP_RIGHT_CORNER_X1, TOP_RIGHT_CORNER_Y1);
    cv::Point2f end(imageSize.width-1, TOP_RIGHT_CORNER_Y2);

    std::vector<cv::Point2f> hull;
    cv::convexHull(std::vector<cv::Point2f>{topRightCorner, start, end}, hull);

    std::vector<cv::Vec4f> filteredSegments;

    for (const cv::Vec4f& segment : segments) {
        cv::Point2f point1(segment[0],segment[1]);
        cv::Point2f point2(segment[2],segment[3]);
        cv::Point2f midPoint = computeMidpoint(segment);

        double result1 = cv::pointPolygonTest(hull, point1, false);  // False = no distance calculation needed
        double result2 = cv::pointPolygonTest(hull, point2, false); 
        double result3 = cv::pointPolygonTest(hull,midPoint,false);

        if(result1<0 && result2<0 && result3 < 0) {
            filteredSegments.push_back(segment);
        }
    }

    return filteredSegments;
}
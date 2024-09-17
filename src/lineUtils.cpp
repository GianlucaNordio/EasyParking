// Davide Molinaroli

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

/**
 * @brief Calculates the angular coefficient (angle) of a line segment in degrees.
 * 
 * This function computes the angle of a line segment defined by two endpoints, expressed in degrees. 
 * The angle is calculated using the arctangent of the ratio of the vertical difference to the horizontal 
 * difference between the endpoints. The result is converted from radians to degrees.
 * 
 * @param segment A `cv::Vec4f` representing the line segment, with coordinates (x1, y1) and (x2, y2).
 * @return The angle of the line segment in degrees, relative to the x-axis.
 */
double getSegmentAngle(const cv::Vec4f& segment) {
    double x1 = segment[0];
    double y1 = segment[1];
    double x2 = segment[2];
    double y2 = segment[3];

    return std::atan((y2 - y1) / (x2 - x1))*180/CV_PI;
}

/**
 * @brief Calculates the length of a line segment.
 * 
 * This function computes the length of a line segment defined by two endpoints. The length is 
 * determined using the Euclidean distance formula, which calculates the distance between the two 
 * endpoints of the segment.
 * 
 * @param segment A `cv::Vec4f` representing the line segment, with coordinates (x1, y1) and (x2, y2).
 * @return The length of the line segment.
 */
double getSegmentLength(const cv::Vec4f& segment) {
    double x1 = segment[0];
    double y1 = segment[1];
    double x2 = segment[2];
    double y2 = segment[3];

    return std::sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

/**
 * @brief Trims the endpoints of two line segments if they intersect.
 * 
 * This function checks if two line segments intersect. If they do, it trims the segments so that their 
 * endpoints are adjusted to the intersection point. For `segment1`, the first endpoint is updated to the 
 * intersection point. For `segment2`, the endpoint closest to the intersection point is trimmed to the 
 * intersection point, while the other endpoint remains unchanged.
 * 
 * @param segment1 A reference to a `cv::Vec4f` representing the first line segment, which will be updated if an intersection occurs.
 * @param segment2 A reference to a `cv::Vec4f` representing the second line segment, which will be updated if an intersection occurs.
 */

void trimIfIntersect(cv::Vec4f& segment1, cv::Vec4f& segment2) {
    cv::Point2f intersection;
    if(segmentsIntersect(segment1, segment2, intersection)) {
        segment1[0] = intersection.x;
        segment1[1] = intersection.y;

        cv::Point2f rightEndpoint(segment2[2], segment2[3]);
        cv::Point2f leftEndpoint(segment2[0], segment2[1]);
        double normLeft = cv::norm(intersection-leftEndpoint);
        double normRight = cv::norm(intersection-rightEndpoint);
        if(normLeft > normRight) {
            segment2[2] = intersection.x;
            segment2[3] = intersection.y;
        }
        else {
            segment2[0] = intersection.x;
            segment2[1] = intersection.y;
        }
    }
}

/**
 * @brief Checks if two line segments intersect and calculates the intersection point if they do.
 * 
 * This function determines if two line segments, each defined by their endpoints, intersect. It calculates 
 * the intersection point using vector algebra. The segments are considered to intersect if the computed 
 * intersection point lies within the bounds of both segments.
 * 
 * @param segment1 A `cv::Vec4f` representing the first line segment with coordinates (x1, y1, x2, y2).
 * @param segment2 A `cv::Vec4f` representing the second line segment with coordinates (x1, y1, x2, y2).
 * @param intersection A reference to a `cv::Point2f` where the intersection point will be stored if the segments intersect.
 * @return `true` if the segments intersect within their bounds and `false` if they do not intersect or are parallel.
 */
bool segmentsIntersect(const cv::Vec4f& segment1, const cv::Vec4f& segment2, cv::Point2f& intersection) {

    // Extract points from the extended segments
    cv::Point2f start1(segment1[0], segment1[1]), end1(segment1[2], segment1[3]);
    cv::Point2f start2(segment2[0], segment2[1]), end2(segment2[2], segment2[3]);

    // Compute direction vectors for both segments
    cv::Vec2f vector1 = cv::Vec2f(end1 - start1);  // Direction of extended_seg1
    cv::Vec2f vector2 = cv::Vec2f(end2 - start2);  // Direction of extended_seg2

    double crossProduct = vector1[0] * vector2[1] - vector1[1] * vector2[0];
    cv::Vec2f vector3 = cv::Vec2f(start2 - start1);
    
    // Check if the lines are parallel
    if (std::fabs(crossProduct) < FLT_EPSILON) {
        return false;  // Parallel lines
    }

    double segment1Param = (vector3[0] * vector2[1] - vector3[1] * vector2[0]) / crossProduct;
    double segment2Param = (vector3[0] * vector1[1] - vector3[1] * vector1[0]) / crossProduct;

    // Check if the intersection happens within the segment bounds
    if (segment1Param >= 0 && segment1Param <= 1 && segment2Param >= 0 && segment2Param <= 1) {
        intersection = start1 + cv::Point2f(vector1[0] * segment1Param, vector1[1] * segment1Param);  // Find the intersection point
        return true;
    }
    return false;
}

/**
 * @brief Extends a line segment in both directions by a specified ratio.
 * 
 * This function extends a line segment, represented by two endpoints, in both directions by a specified 
 * extension ratio. The extension is calculated based on the length of the segment and is applied to both 
 * ends of the segment. The function computes the direction vector of the segment, normalizes it, and then 
 * extends the segment accordingly.
 * 
 * @param segment A `cv::Vec4f` representing the line segment, with coordinates (x1, y1, x2, y2).
 * @param extensionRatio The ratio by which to extend the segment relative to its length (e.g., 0.25 for 25% extension).
 * @return A `cv::Vec4f` representing the extended line segment, with updated coordinates for the new endpoints.
 */
cv::Vec4f extendSegment(const cv::Vec4f& segment, double extensionRatio) {
    cv::Point2f start(segment[0], segment[1]), end(segment[2], segment[3]);

    // Compute direction vector of the segment
    cv::Vec2f direction = cv::Vec2f(end - start);
    double length = getSegmentLength(segment);
    
    // Normalize the direction vector to unit length
    cv::Vec2f directionNormalized = direction / length;

    // Compute the extension length (25% of the segment length)
    double extensionLength = length * extensionRatio;

    // Extend in both directions by converting to cv::Point2f for vector arithmetic
    cv::Point2f extendedStart = start - cv::Point2f(directionNormalized[0], directionNormalized[1]) * extensionLength;
    cv::Point2f extendedEnd = end + cv::Point2f(directionNormalized[0], directionNormalized[1]) * extensionLength;

    // Return the new extended segment
    return cv::Vec4f(extendedStart.x, extendedStart.y, extendedEnd.x, extendedEnd.y);
}

/**
 * @brief Computes the average of a vector of double values.
 * 
 * This function calculates the average (mean) value of a vector of `double` numbers. It sums all the 
 * elements in the vector and divides by the number of elements to obtain the average. If the vector 
 * is empty, the function returns 0.0 to avoid division by zero.
 * 
 * @param data A vector of `double` values for which the average is to be computed.
 * @return The average of the values in the vector, or 0.0 if the vector is empty.
 */
double computeAvg(std::vector<double>& data) {
    if (data.empty()) return 0.0;
    double const count = static_cast<double>(data.size());
    return std::reduce(data.begin(), data.end()) / count;
}
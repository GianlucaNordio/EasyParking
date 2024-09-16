#include "parkingSpotUtils.hpp"

/**
 * @brief Performs Non-Maximum Suppression (NMS) on a set of rotated rectangles based on their overlap and size.
 * 
 * This function processes a list of rotated rectangles and removes redundant overlapping rectangles 
 * based on a specified intersection area threshold. The decision on which rectangle to remove is 
 * made based on their area size, and can be configured to keep either the smallest or largest rectangle.
 *
 * @param rects A vector of cv::RotatedRect objects representing the bounding boxes to process.
 * @param elementsToRemove A vector of cv::RotatedRect objects that will store the rectangles marked for removal.
 * @param threshold A double representing the normalized intersection area threshold above which two rectangles 
 * are considered overlapping.
 * @param keepSmallest A boolean flag that indicates whether to keep the smallest rectangle (true) 
 * or the largest rectangle (false) when two rectangles overlap.
 *
 * @note The function assumes that the rects vector contains unique rectangles. 
 * Rectangles whose centers are exactly the same will not be compared.
 */
void nonMaximumSuppression(std::vector<cv::RotatedRect> &rects, std::vector<cv::RotatedRect> &elementsToRemove, double threshold, bool keepSmallest) {
    for (const cv::RotatedRect& rect1 : rects) {
        for (const cv::RotatedRect& rect2 : rects) {
            if (!(rect1.center.x == rect2.center.x && rect1.center.y == rect2.center.y) && (computeIntersectionAreaNormalized(rect1, rect2) > threshold)) {
                if (keepSmallest ? rect1.size.area() < rect2.size.area() : rect1.size.area() > rect2.size.area()){
                    elementsToRemove.push_back(rect2);
                } else {
                    elementsToRemove.push_back(rect1);
                }
            }
        }
    }
}

/**
 * @brief Computes the normalized intersection area between two rotated rectangles.
 *
 * This function calculates the area of intersection between two rotated rectangles (`rect1` and `rect2`) and
 * normalizes it by dividing by the smaller of the two rectangle areas. The normalization ensures that the 
 * intersection area is represented as a ratio, where 1 means complete overlap and 0 means no overlap.
 *
 * The function first extracts the corner points of both rectangles, then uses OpenCV's 
 * `cv::intersectConvexConvex` to find the intersection polygon. The area of this intersection is divided by the 
 * smaller of the two original rectangle areas.
 *
 * @param rect1 The first rotated rectangle.
 * @param rect2 The second rotated rectangle.
 * @return The normalized intersection area as a ratio in the range [0, 1], where 1 indicates complete overlap
 *         and 0 indicates no intersection.
 */
double computeIntersectionAreaNormalized(const cv::RotatedRect& rect1, const cv::RotatedRect& rect2) {
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
    double intersectionArea = cv::intersectConvexConvex(points1, points2, intersection) / std::min(area1, area2);

    return intersectionArea;
}

/**
 * @brief Finds the iterator to an element in a vector of rotated rectangles based on the center coordinates.
 *
 * This function searches through a vector of `cv::RotatedRect` and returns an iterator to the first element
 * that has the same center coordinates (`center.x` and `center.y`) as the specified `elem`. If no such element
 * is found, the function returns `rects.cend()`, indicating the end of the vector.
 *
 * @param rects A constant reference to a vector of rotated rectangles (`cv::RotatedRect`).
 * @param elem The rotated rectangle whose center coordinates will be used for comparison.
 * @return A constant iterator to the found element in the vector, or `rects.cend()` if the element is not found.
 */
std::vector<cv::RotatedRect>::const_iterator elementIterator(const std::vector<cv::RotatedRect>& rects, const cv::RotatedRect& elem){
    for (auto it = rects.cbegin(); it != rects.cend(); ++it) {
        if (it->center.x == elem.center.x &&
            it->center.y == elem.center.y) 
        {
            return it;
        }
    }
    return rects.cend(); 
}

/**
 * @brief Converts a cv::RotatedRect into a line segment represented by its two midpoints.
 *
 * This function takes a rotated rectangle (cv::RotatedRect) and calculates the midpoints
 * of the two longest opposite edges. It then returns a line segment that connects these
 * midpoints as a vector of four floating point values (x1, y1, x2, y2), where (x1, y1)
 * is the midpoint of one edge and (x2, y2) is the midpoint of the opposite edge.
 *
 * The function identifies the two longest opposite edges of the rectangle and calculates
 * their midpoints, which are used to define the line segment.
 *
 * @param rect The rotated rectangle (cv::RotatedRect) to convert into a line.
 * 
 * @return cv::Vec4f A vector of four floats representing the line segment: (x1, y1, x2, y2).
 */
cv::Vec4f convertRectToLine(const cv::RotatedRect& rect) {
    cv::Point2f points[4];
    rect.points(points);

    // Calculate the length of the edges
    double length1 = cv::norm(points[0] - points[1]);
    double length2 = cv::norm(points[1] - points[2]);
    double length3 = cv::norm(points[2] - points[3]);
    double length4 = cv::norm(points[3] - points[0]);

    // The longest two opposite edges define the line
    double maxLength1 = std::max(length1, length3);
    double maxLength2 = std::max(length2, length4);

    // Midpoints of the longest edges
    cv::Point2f midPoint1, midPoint2;

    if (maxLength1 < maxLength2) {
        // Use points 0->1 and 2->3 (longest edge pair)
        midPoint1 = (points[0] + points[1]) * 0.5f;
        midPoint2 = (points[2] + points[3]) * 0.5f;
    } else {
        // Use points 1->2 and 3->0 (other longest edge pair)
        midPoint1 = (points[1] + points[2]) * 0.5f;
        midPoint2 = (points[3] + points[0]) * 0.5f;
    }

    // Return the line segment as a vector of 4 floats (x1, y1, x2, y2)
    return cv::Vec4f(midPoint1.x, midPoint1.y, midPoint2.x, midPoint2.y);
}

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
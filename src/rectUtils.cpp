// Davide Molinaroli

#include "rectUtils.hpp"

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
 * @brief Splits a rotated rectangle into two smaller rectangles and shifts them apart.
 * 
 * This function takes a `cv::RotatedRect` and splits it into two smaller rectangles, each 
 * having half the height of the original rectangle. The splitting occurs along the longest side 
 * of the original rectangle (between vertices 0 and 1). The midpoints of these sides are calculated, 
 * and the resulting smaller rectangles are shifted apart along the x and y axes, based on the angle of the original rectangle.
 * This is done because if both parts (obtained by splitting a rectangle), when slightly shifted and rotated, overlap with other rectangles, 
 * then we are between two rows of parking spots, which is not possible
 * 
 * The shift direction is determined by the rectangle's angle, adjusted by an optional offset 
 * defined by `SPLIT_DIRECTION_OFFSET`, and the shift amount is proportional to the original rectangle's width.
 * 
 * @param rect The input rotated rectangle to be split and shifted.
 * @return A pair of `cv::RotatedRect` objects, representing the two split and shifted parts of the original rectangle.
 */
std::pair<cv::RotatedRect, cv::RotatedRect> splitAndShiftRotatedRect(const cv::RotatedRect& rect) {
    cv::Point2f vertices[4];
    rect.points(vertices);
    double shiftAmount = rect.size.width;

    // Find the midpoint along the longest side (between vertices[0] and vertices[1])
    cv::Point2f midPoint1 = (vertices[0] + vertices[1]) * 0.5;
    cv::Point2f midPoint2 = (vertices[2] + vertices[3]) * 0.5;

    // Calculate the shift direction based on the angle of the rectangle
    double angleRad = (rect.angle + SPLIT_DIRECTION_OFFSET) * CV_PI / 180.0;  // Convert angle to radians
    cv::Point2f shiftVectorX(shiftAmount * std::cos(-angleRad), shiftAmount * std::sin(-angleRad));  // Shift in x-direction
    cv::Point2f shiftVectorY(-shiftAmount * std::sin(-angleRad), shiftAmount * std::cos(-angleRad)); // Shift in y-direction

    // Shift the midpoints along the x and y axes
    cv::Point2f shiftedCenter1 = midPoint1 - shiftVectorX - shiftVectorY;
    cv::Point2f shiftedCenter2 = midPoint2 + shiftVectorX + shiftVectorY;

    // Create two new rotated rects, each with half the original width, and shifted
    cv::RotatedRect rectPart1(shiftedCenter1, cv::Size2f(rect.size.width, rect.size.height/2), rect.angle + SPLIT_DIRECTION_OFFSET);
    cv::RotatedRect rectPart2(shiftedCenter2, cv::Size2f(rect.size.width , rect.size.height/2), rect.angle + SPLIT_DIRECTION_OFFSET);
    
    return std::make_pair(rectPart1, rectPart2);
}

/**
 * @brief Resolves overlaps between two sets of rotated rectangles by shifting rectangles in the second set.
 * 
 * This function iterates through two vectors of rotated rectangles and checks for overlaps between rectangles 
 * in `vector1` and `vector2`. If an overlap is detected (based on a normalized intersection area), the function 
 * shifts the overlapping rectangle from `vector2` along its longest axis by a specified shift amount until the overlap 
 * is resolved.
 * 
 * The overlap is resolved when the normalized intersection area between the two rectangles becomes less than the 
 * `RESOLVE_OVERLAP_THRESHOLD`. The `shiftAlongLongestAxis` function is used to move the rectangle.
 * 
 * @param vector1 A vector of `cv::RotatedRect` objects representing the first set of rectangles.
 * @param vector2 A vector of `cv::RotatedRect` objects representing the second set of rectangles to be shifted if overlaps occur.
 * @param shiftAmount The distance by which overlapping rectangles in `vector2` are shifted along their longest axis.
 */
void resolveOverlaps(std::vector<cv::RotatedRect>& vector1, std::vector<cv::RotatedRect>& vector2, double shiftAmount) {
    for (cv::RotatedRect& rect1 : vector1) {
        for (cv::RotatedRect& rect2 : vector2) {

            // Check if the two rectangles overlap
            while (computeIntersectionAreaNormalized(rect1, rect2) > RESOLVE_OVERLAP_THRESHOLD) {
                
                // Shift rect2 along its longest axis until it no longer overlaps
                rect2 = shiftAlongLongestAxis(rect2, shiftAmount, false);
            }
        }
    }
}

/**
 * @brief Shifts a rotated rectangle along its longest axis by a specified amount.
 * 
 * This function calculates the direction of the longest axis of a `cv::RotatedRect` and shifts 
 * the rectangle's center along this axis by the specified `shiftAmount`. The direction of the shift can 
 * be inverted if `invertDirection` is set to `true`. The shifted rectangle retains its original size 
 * and rotation angle.
 * 
 * The longest axis is determined by comparing the distances between the rectangle's vertices. 
 * The shift is applied by normalizing the direction vector along the longest axis and moving the 
 * rectangle's center by the specified amount in that direction.
 * 
 * @param rect The input rotated rectangle to be shifted.
 * @param shiftAmount The amount by which to shift the rectangle along its longest axis.
 * @param invertDirection A boolean flag indicating whether to shift in the opposite direction of the axis.
 * @return A new `cv::RotatedRect` with the shifted center, keeping the original size and rotation angle.
 */
cv::RotatedRect shiftAlongLongestAxis(const cv::RotatedRect& rect, double shiftAmount, bool invertDirection) {
    // Find the longer dimension of the rectangle
    cv::Point2f vertices[4];
    rect.points(vertices);

    // Compute the longest axis direction
    cv::Point2f axis = (cv::norm(vertices[0] - vertices[1]) > cv::norm(vertices[1] - vertices[2])) ?
                        (vertices[1] - vertices[0]) : (vertices[2] - vertices[1]);

    // Normalize the axis vector to shift along its direction
    cv::Point2f normalizedAxis = axis / cv::norm(axis);

    // Shift the center of the rotated rect along this axis
    cv::Point2f newCenter = rect.center + shiftAmount * (invertDirection ? -normalizedAxis: normalizedAxis);

    // Return a new rotated rect with the updated center
    return cv::RotatedRect(newCenter, rect.size, rect.angle);
}

/**
 * @brief Checks if a rotated rectangle is isolated, i.e., it does not overlap with any other rectangles in a given list.
 * 
 * This function determines if a `cv::RotatedRect` is isolated by first scaling the rectangle using 
 * the `scaleRotatedRect` function with a predefined `SCALE_FACTOR`. It then checks for overlaps 
 * between the scaled rectangle and each rectangle in a provided vector, `rects`. If any rectangle in 
 * the vector overlaps with the scaled rectangle (excluding overlap with the rectangle itself), the function 
 * returns `false`, indicating the rectangle is not isolated. Otherwise, it returns `true`.
 * 
 * @param rect The rotated rectangle to be checked for isolation.
 * @param rects A vector of `cv::RotatedRect` objects representing other rectangles to compare against.
 * @return `true` if the rectangle is isolated (i.e., does not overlap with any other rectangles), 
 *         `false` if there is any overlap with other rectangles in the list.
 */
bool isAlone(cv::RotatedRect rect, std::vector<cv::RotatedRect> rects) {
    cv::RotatedRect extended = scaleRotatedRect(rect, SCALE_FACTOR);
    for(const cv::RotatedRect otherRect:rects) {
        if(otherRect.center != rect.center && areRectsOverlapping(extended, otherRect)) {
            return false;
        }
    }

    return true;
}

/**
 * @brief Scales a rotated rectangle by a specified factor.
 * 
 * This function scales the size (width and height) of a `cv::RotatedRect` by a given `scaleFactor`. 
 * The function creates and returns a new `cv::RotatedRect` with the updated size while keeping 
 * the original center and rotation angle unchanged.
 * 
 * @param rect The input rotated rectangle to be scaled.
 * @param scaleFactor The factor by which to scale the rectangle's size. A value greater than 1 
 *                    increases the size, while a value less than 1 decreases it.
 * @return A new `cv::RotatedRect` with the scaled size, same center, and same rotation angle as the original.
 */
cv::RotatedRect scaleRotatedRect(const cv::RotatedRect& rect, double scaleFactor) {
    // Scale the size (width and height) of the rotated rect
    cv::Size2f newSize(rect.size.width * scaleFactor, rect.size.height * scaleFactor);

    // Create a new rotated rect with the scaled size
    return cv::RotatedRect(rect.center, newSize, rect.angle);
}

/**
 * @brief Determines whether two rotated rectangles overlap.
 * 
 * This function checks if two `cv::RotatedRect` objects overlap by using the OpenCV function 
 * `cv::rotatedRectangleIntersection`, which computes the intersection points between the rectangles. 
 * The function returns `true` if the rectangles fully or partially overlap, and `false` otherwise.
 * 
 * The possible intersection statuses are:
 * - `cv::INTERSECT_FULL`: The rectangles fully overlap.
 * - `cv::INTERSECT_PARTIAL`: The rectangles partially overlap.
 * 
 * @param rect1 The first rotated rectangle.
 * @param rect2 The second rotated rectangle.
 * @return `true` if the rectangles fully or partially overlap, `false` otherwise.
 */
bool areRectsOverlapping(const cv::RotatedRect& rect1, const cv::RotatedRect& rect2) {
    std::vector<cv::Point2f> intersectionPoints;
    int intersectionStatus = cv::rotatedRectangleIntersection(rect1, rect2, intersectionPoints);
    return intersectionStatus == cv::INTERSECT_FULL || intersectionStatus == cv::INTERSECT_PARTIAL;
}

/**
 * @brief Merges overlapping and aligned rotated rectangles into a single rectangle.
 * 
 * This function takes a vector of `cv::RotatedRect` objects and merges those that overlap and are aligned 
 * within a specified angular tolerance. The merging process groups overlapping and aligned rectangles, 
 * then creates a single bounding box that encompasses all the grouped rectangles. The resulting merged 
 * rectangles are returned in a new vector.
 * 
 * The function uses the following steps:
 * - Iterates through the list of rectangles to find groups of overlapping and aligned rectangles.
 * - For each group, collects the points of all rectangles in the group.
 * - Computes a new bounding `cv::RotatedRect` for the group of points using `cv::minAreaRect`.
 * 
 * @param rects A vector of `cv::RotatedRect` objects to be merged.
 * @return A vector of merged `cv::RotatedRect` objects, where each represents the bounding box of a group of overlapping 
 *         and aligned rectangles.
 */
std::vector<cv::RotatedRect> mergeOverlappingRects(std::vector<cv::RotatedRect>& rects) {
    std::vector<cv::RotatedRect> mergedRects;
    std::vector<bool> merged(rects.size(), false);  // Track which rects have been merged

    for (size_t i = 0; i < rects.size(); ++i) {
        if (merged[i]) continue;  // Skip already merged rects

        // Start a new group of merged rects
        std::vector<cv::Point2f> groupPoints;
        cv::Point2f points[4];
        rects[i].points(points);
        groupPoints.insert(groupPoints.end(), points, points + 4);
        merged[i] = true;

        // Check for overlap and alignment with other rects
        for (size_t j = i + 1; j < rects.size(); ++j) {
            if (!merged[j] && areRectsOverlapping(rects[i], rects[j]) && areRectsAligned(rects[i], rects[j], ANGLE_TOLERANCE)) {
                
                // Merge the overlapping and aligned rect
                rects[j].points(points);
                groupPoints.insert(groupPoints.end(), points, points + 4);
                merged[j] = true;
            }
        }

        // Create a single bounding box from the merged group points
        cv::RotatedRect mergedRect = cv::minAreaRect(groupPoints);
        mergedRects.push_back(mergedRect);
    }

    return mergedRects;
}

/**
 * @brief Checks if two rotated rectangles are aligned within a specified angular tolerance.
 * 
 * This function determines if two `cv::RotatedRect` objects have similar orientations by comparing 
 * their rotation angles. The rectangles are considered aligned if the absolute difference between their 
 * angles is less than or equal to a specified `angleTolerance`.
 * 
 * @param rect1 The first rotated rectangle.
 * @param rect2 The second rotated rectangle.
 * @param angleTolerance The maximum allowable difference in rotation angles for the rectangles to be considered aligned.
 * @return `true` if the absolute difference between the angles of the two rectangles is less than or equal to `angleTolerance`,
 *         `false` otherwise.
 */
bool areRectsAligned(const cv::RotatedRect& rect1, const cv::RotatedRect& rect2, double angleTolerance) {
    return std::abs(rect1.angle - rect2.angle) <= angleTolerance;
}

/**
 * @brief Filters out rotated rectangles based on their overlap with surrounding rectangles.
 * 
 * This function processes a list of rotated rectangles (`rects1`) by splitting each rectangle into two 
 * parts along its longest direction and scaling both parts. It then checks if these scaled parts overlap 
 * with any rectangle in another list (`rects2`). If both parts of a rectangle from `rects1` overlap with 
 * at least one rectangle in `rects2`, the original rectangle is discarded. Rectangles with an area smaller 
 * than a defined minimum (`MIN_AREA`) are also discarded.
 * 
 * @param rects1 A vector of `cv::RotatedRect` objects to be filtered based on their overlap with `rects2`.
 * @param rects2 A vector of `cv::RotatedRect` objects used to check for overlaps with the rectangles in `rects1`.
 * @param image The image associated with the rectangles (not used directly in the function but may be for context).
 * @return A vector of `cv::RotatedRect` objects from `rects1` that do not have both parts overlapping with any 
 *         rectangle in `rects2` and have an area greater than `MIN_AREA`.
 */
std::vector<cv::RotatedRect> filterBySurrounding(const std::vector<cv::RotatedRect>& rects1, const std::vector<cv::RotatedRect>& rects2, cv::Mat image) {
    std::vector<cv::RotatedRect> filteredRects;

    for (const cv::RotatedRect& rect1 : rects1) {
        // Split rect1 into two equal parts along its longest direction
        std::pair<cv::RotatedRect, cv::RotatedRect> result = splitAndShiftRotatedRect(rect1);
        cv::RotatedRect rectPart1 = result.first;
        cv::RotatedRect rectPart2 = result.second;
        
        // Scale both parts
        rectPart1 = scaleRotatedRect(rectPart1, 1.25);
        rectPart2 = scaleRotatedRect(rectPart2, 1.25);
        
        bool part1Overlap = false, part2Overlap = false;

        // Check if both parts overlap with any rect in rects2
        for (const cv::RotatedRect& rect2 : rects2) {
            if (computeIntersectionAreaNormalized(rectPart1, rect2) > 0) {
                part1Overlap = true;
            }
            if (computeIntersectionAreaNormalized(rectPart2, rect2) > 0) {
                part2Overlap = true;
            }
            // If both parts overlap with at least one rect, we can discard rect1
            if (part1Overlap && part2Overlap) {
                break;
            }
        }

        // Only keep rect1 if not both parts overlap with any rect in rects2
        if (!(part1Overlap && part2Overlap) && rect1.size.area() > MIN_AREA) {
            filteredRects.push_back(rect1);
        }
    }

    return filteredRects;
}

/**
 * @brief Computes the median of a vector of double values.
 * 
 * This function calculates the median value of a vector of `double` numbers. It first sorts the vector 
 * and then determines the median based on whether the number of elements is even or odd:
 * - For an odd number of elements, the median is the middle value.
 * - For an even number of elements, the median is the average of the two middle values.
 * If the vector is empty, the function returns 0.0 to handle the case of no data.
 * 
 * @param data A vector of `double` values for which the median is to be computed.
 * @return The median value of the vector, or 0.0 if the vector is empty.
 */
double computeMedian(std::vector<double>& data) {
    if (data.empty()) return 0.0;
    std::sort(data.begin(), data.end());
    size_t n = data.size();
    if (n % 2 == 0) {
        return (data[n / 2 - 1] + data[n / 2]) / 2.0;
    } else {
        return data[n / 2];
    }
}
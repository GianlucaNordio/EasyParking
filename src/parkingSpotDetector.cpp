// Davide Molinaroli

#include "parkingSpotDetector.hpp"

/**
 * @brief Detects parking spots in a sequence of images, applies Non-Maximum Suppression (NMS) 
 * to remove overlapping or redundant detections, and stores the filtered spots.
 *
 * This function processes each image in the provided vector of images to detect potential parking spots.
 * For each image, it identifies the parking spots using the `detectParkingSpotInImage` function and 
 * stores the spots in the `baseSequenceParkingSpots`. The coordinates of all the detected spots 
 * are accumulated into a vector and then passed to the Non-Maximum Suppression (NMS) function to 
 * filter out redundant or overlapping spots. Finally, the best filtered parking spots are stored in 
 * the `bestParkingSpots` vector.
 *
 * @param images A vector of OpenCV Mat objects, where each Mat represents an image.
 * @param bestParkingSpots A reference to a vector where the best parking spots (after NMS) will be stored.
 * @param baseSequenceParkingSpots A reference to a vector of vectors, where parking spots for each image are stored.
 */
void detectParkingSpots(const std::vector<cv::Mat>& images, std::vector<ParkingSpot>& bestParkingSpots, std::vector<::std::vector<ParkingSpot>>& baseSequenceParkingSpots) {

    std::vector<cv::RotatedRect> allRectsFound;
    for(const cv::Mat& image : images) {
        // Find parking spots for each image separately
        std::vector<ParkingSpot> parkingSpotsImage;
        detectParkingSpotInImage(image, parkingSpotsImage);
        baseSequenceParkingSpots.push_back(parkingSpotsImage);
        for(ParkingSpot spot : parkingSpotsImage) {
            allRectsFound.push_back(spot.rect);
        }
    }

    std::vector<cv::RotatedRect> elementToRemove;
    nonMaximumSuppression(allRectsFound, elementToRemove, NON_MAXIMUM_SUPPRESSION_THRESHOLD, true);

    // Remove the elements determined by NMS filtering
    for (cv::RotatedRect element : elementToRemove) {
        std::vector<cv::RotatedRect>::const_iterator iterator = elementIterator(allRectsFound, element);
        if (iterator != allRectsFound.cend()) {
            allRectsFound.erase(iterator);
        }
    }

    int count = 0;

    for (cv::RotatedRect& rect : allRectsFound) {
        bestParkingSpots.push_back(ParkingSpot(count, 1 , false, rect));
        count++;
    }
}

/**
 * @brief Detects parking spots in a single image by analyzing line segments and applying template matching and 
 * Non-Maximum Suppression (NMS) to identify valid parking spot regions.
 *
 * This function performs a multi-step approach to detect parking spots within the input image. First, it preprocesses 
 * the image to find white lines. After detecting line segments, it filters out short segments and segments near 
 * the top-right of the image. The function calculates the angles and lengths of remaining segments, performs 
 * multi-scale template matching using these average values, and merges overlapping bounding boxes. Non-Maximum 
 * Suppression (NMS) is applied to further refine the detected regions. Finally, filtered bounding boxes are converted 
 * into `ParkingSpot` objects, which are returned via the `parkingSpots` vector.
 *
 * @param image The input image as an OpenCV Mat object in which parking spots will be detected.
 * @param parkingSpots A reference to a vector where detected parking spots will be stored.
 */
void detectParkingSpotInImage(const cv::Mat& image, std::vector<ParkingSpot>& parkingSpots) {
	std::vector<cv::RotatedRect> spots;
    cv::Mat preprocessed = preprocessFindWhiteLines(image);
    
	cv::Ptr<cv::LineSegmentDetector > lineSegmentDetector = cv::createLineSegmentDetector();
    std::vector<cv::Vec4f> lineSegment;
    lineSegmentDetector->detect(preprocessed,lineSegment);
    std::vector<cv::Vec4f> segments;

	//Reject short lines
    for(int i = 0; i < lineSegment.size(); i++) {                                                                                                                         //150
        if(getSegmentLength(lineSegment[i]) > MIN_SEGMENT_LENGTH) {
            segments.push_back(lineSegment[i]);
        }
    }

    std::vector<cv::Vec4f> filteredSegments = filterSegmentsNearTopRight(segments, cv::Size(image.cols, image.rows));

    std::vector<double> positiveAngles;
    std::vector<double> negativeAngles;
    std::vector<double> positiveLengths;
    std::vector<double> negativeLengths;

    for (const cv::Vec4f& segment : filteredSegments) {
        // Calculate the angle with respect to the x-axis
        double angle = getSegmentAngle(segment);

        // Calculate the length of the segment (line width)
        double length = getSegmentLength(segment);

        // Categorize and store the angle and length
        if (angle > 0) {
            positiveAngles.push_back(angle);
            positiveLengths.push_back(length);

        } else if (angle < 0) {
            negativeAngles.push_back(angle);
            negativeLengths.push_back(length);
        }
    }

    double avgPositiveAngle = computeAvg(positiveAngles);
    double avgNegativeAngle = computeAvg(negativeAngles);
    double avgPositiveWidth = computeAvg(positiveLengths);
    double avgNegativeWidth = computeAvg(negativeLengths);

    preprocessed = preprocessFindParkingLines(image);
 
    std::vector<cv::RotatedRect> positiveRects;
    std::vector<cv::RotatedRect> negativeRects;
    
    multiRotationTemplateMatching(preprocessed, avgPositiveWidth, avgPositiveAngle, TEMPLATE_HEIGHT_MULTISCALE, TEMPLATE_MATCHING_SCALE_TEMPLATE_POSITIVE, TEMPLATE_MATCHING_SCALE_RECT_POSITIVE, TEMPLATE_MATCHING_THRESHOLD, POSITIVE_ANGLE_OFFSET, positiveRects, true);
    multiRotationTemplateMatching(preprocessed, avgNegativeWidth, avgNegativeAngle, TEMPLATE_HEIGHT_MULTISCALE, TEMPLATE_MATCHING_SCALE_TEMPLATE_NEGATIVE, TEMPLATE_MATCHING_SCALE_RECT_NEGATIVE, TEMPLATE_MATCHING_THRESHOLD, NEGATIVE_ANGLE_OFFSET, negativeRects, false);

    std::vector<cv::RotatedRect> mergedPositiveRects = mergeOverlappingRects(positiveRects);
    std::vector<cv::RotatedRect> mergedNegativeRects = mergeOverlappingRects(negativeRects);

    std::vector<cv::Vec4f> segmentsPositive;
    std::vector<cv::Vec4f> segmentsNegative;

    // Loop through all bounding boxes
    for (const cv::RotatedRect& rect : mergedPositiveRects) {
        cv::Vec4f lineSegment = convertRectToLine(rect);
        segmentsPositive.push_back(lineSegment);
    }

    // Loop through all bounding boxes
    for (const cv::RotatedRect& rect : mergedNegativeRects) {
        cv::Vec4f lineSegment = convertRectToLine(rect);
        segmentsNegative.push_back(lineSegment);
    }

    std::vector<cv::Vec4f> noTopRightNegative = filterSegmentsNearTopRight(segmentsNegative, cv::Size(image.cols, image.rows));
    std::vector<cv::Vec4f> noTopRightPositive = filterSegmentsNearTopRight(segmentsPositive, cv::Size(image.cols, image.rows));

    double positiveDistanceThreshold;
    double negativeDistanceThreshold;

    positiveDistanceThreshold = avgPositiveWidth * MIDPOINT_DISTANCE_FACTOR;
    negativeDistanceThreshold = avgNegativeWidth * MIDPOINT_DISTANCE_FACTOR;

    std::vector<cv::Vec4f> filteredSegmentNegative = filterCloseSegments(noTopRightNegative, negativeDistanceThreshold);
    std::vector<cv::Vec4f> filteredSegmentPositive = filterCloseSegments(noTopRightPositive, positiveDistanceThreshold);

    for(cv::Vec4f& negativeLine: filteredSegmentNegative) {
        for(cv::Vec4f positiveLine: filteredSegmentPositive) {
            trimIfIntersect(negativeLine, positiveLine);
        } 
    }

    // Process the segments
    std::vector<cv::RotatedRect> positiveRotatedRects = buildRotatedRectsFromSegments(filteredSegmentPositive);
    std::vector<cv::RotatedRect> negativeRotatedRects = buildRotatedRectsFromSegments(filteredSegmentNegative);

    // Apply NMS filtering
    std::vector<cv::RotatedRect> positiveElementsToRemove;
    std::vector<cv::RotatedRect> negativeElementsToRemove;
    nonMaximumSuppression(positiveRotatedRects, positiveElementsToRemove, NON_MAXIMUM_SUPPRESSION_THRESHOLD, false); 
    nonMaximumSuppression(negativeRotatedRects, negativeElementsToRemove, NON_MAXIMUM_SUPPRESSION_THRESHOLD_LOW, false);

    // Remove the elements determined by NMS filtering
    for (cv::RotatedRect element : positiveElementsToRemove) {
        std::vector<cv::RotatedRect>::const_iterator iterator = elementIterator(positiveRotatedRects, element);
        if (iterator != positiveRotatedRects.cend()) {
            positiveRotatedRects.erase(iterator);
        }
    }

    // Remove the elements determined by NMS filtering
    for (cv::RotatedRect element : negativeElementsToRemove) {
        std::vector<cv::RotatedRect>::const_iterator iterator = elementIterator(negativeRotatedRects, element);
        if (iterator != negativeRotatedRects.cend()) {
            negativeRotatedRects.erase(iterator);
        }
    }

    std::vector filteredRects = filterBySurrounding(negativeRotatedRects, positiveRotatedRects, image);
    std::vector<double> areas;
    double medianArea;

    for (const cv::RotatedRect& rect : positiveRotatedRects) {
        if(rect.size.area() < MIN_AREA) continue;
        areas.push_back(rect.size.area());
    }

    medianArea = computeMedian(areas);
    areas.clear();

    std::vector<cv::RotatedRect> removeBigAndSmallPositive;
    std::vector<cv::RotatedRect> allRectsFound;

    for (const cv::RotatedRect& rect : positiveRotatedRects) {
        if(rect.size.area() > medianArea * MEDIAN_AREA_MIN_FACTOR && rect.size.area() < medianArea * MEDIAN_AREA_MAX_FACTOR) {
            removeBigAndSmallPositive.push_back(rect);
        }
    }

    for (const cv::RotatedRect& rect : filteredRects) {
        if(rect.size.area() < MIN_AREA) continue;
        areas.push_back(rect.size.area());
    }

    std::vector<cv::RotatedRect> removeBigAndSmallNegative;
    medianArea = computeMedian(areas);

    for (const cv::RotatedRect& rect : filteredRects) {
        if(rect.size.area() > medianArea * MEDIAN_AREA_MIN_FACTOR && rect.size.area() < medianArea * MEDIAN_AREA_MAX_FACTOR) {
            removeBigAndSmallNegative.push_back(rect);
            allRectsFound.push_back(rect);
        }
    }

    // Resolve overlaps between vector1 and vector2
    resolveOverlaps(removeBigAndSmallNegative, removeBigAndSmallPositive, SHIFT_AMOUNT);

    // re-do nms after overlaps are resolved
    std::vector<cv::RotatedRect> elementsToRemoveFinal;
    nonMaximumSuppression(removeBigAndSmallPositive, elementsToRemoveFinal, NON_MAXIMUM_SUPPRESSION_THRESHOLD_HIGH, false);

    // Remove the elements determined by NMS filtering
    for (cv::RotatedRect element : elementsToRemoveFinal) {
        std::vector<cv::RotatedRect>::const_iterator iterator = elementIterator(removeBigAndSmallPositive, element);
        if (iterator != removeBigAndSmallPositive.cend()) {
            removeBigAndSmallPositive.erase(iterator);
        }
    }

    for (const cv::RotatedRect& rect : removeBigAndSmallPositive) {
        if(rect.size.area() > MIN_AREA) { // the if is needed because removing with the iterator produces rects with zero area
            allRectsFound.push_back(rect);
        }
    }

    std::vector<cv::RotatedRect> finalRects;
    for(const cv::RotatedRect& rect : allRectsFound) {
        if(!isAlone(rect, allRectsFound)) {
            finalRects.push_back(rect);
            cv::Point2f vertices[4];
            rect.points(vertices);
        }
    }

	for (const cv::RotatedRect& rect : finalRects) {
        parkingSpots.push_back(ParkingSpot(NO_ID, CONFIDENCE, false, rect));
    }
}

/**
 * @brief Constructs rotated rectangles from a vector of segments.
 * 
 * This function processes a vector of segments to build rotated rectangles. For each segment in the input vector:
 * 1. It calls the `buildRotatedRectFromPerpendicular` function to create a rotated rectangle based on the segment.
 * 2. It checks if the resulting rotated rectangle has a size area greater than a predefined minimum area.
 * 3. Only rectangles meeting the area criterion are added to the result vector.
 * 
 * @param segments A vector of `cv::Vec4f` representing the segments, where each `cv::Vec4f` contains coordinates of a segment (x1, y1, x2, y2).
 * @return A vector of `cv::RotatedRect` representing the constructed rotated rectangles that meet the size area requirement.
 */
std::vector<cv::RotatedRect> buildRotatedRectsFromSegments(const std::vector<cv::Vec4f>& segments) {
    std::vector<cv::RotatedRect> rotatedRects;

    // Iterate over each segment
    for (const cv::Vec4f& segment : segments) {
        // Process each segment and build the rotated rectangle
        cv::RotatedRect rotatedRect = buildRotatedRectFromPerpendicular(segment, segments);
        if(rotatedRect.size.area()> MIN_AREA)
            rotatedRects.push_back(rotatedRect);
    }

    return rotatedRects;
}

/**
 * @brief Builds a rotated rectangle based on a perpendicular segment and other segments.
 * 
 * This function constructs a rotated rectangle from a given segment by extending the segment and finding
 * intersections with other segments. The process involves:
 * 1. Calculating the perpendicular direction to the original segment.
 * 2. Extending the segment to a certain length.
 * 3. Checking for intersections between the extended perpendicular segment and other segments.
 * 4. Using the closest intersection to define the rotated rectangle.
 * 
 * The function also adjusts the center of the resulting rotated rectangle by a predefined shift.
 * 
 * @param segment A `cv::Vec4f` representing the original segment, with coordinates (x1, y1, x2, y2).
 * @param segments A vector of `cv::Vec4f` representing other segments to check for intersections.
 * @return A `cv::RotatedRect` representing the constructed rotated rectangle, or an empty `cv::RotatedRect` if no valid intersection is found.
 */
cv::RotatedRect buildRotatedRectFromPerpendicular(const cv::Vec4f& segment, const std::vector<cv::Vec4f>& segments) {
    // Rightmost endpoint of the original segment
    cv::Point2f rightEndpoint(segment[2], segment[3]);
    cv::Point2f leftEndpoint(segment[0], segment[1]);

    // Compute the direction vector of the original segment
    cv::Vec2f direction = cv::Vec2f(rightEndpoint - leftEndpoint);
    double length = std::sqrt(direction[0] * direction[0] + direction[1] * direction[1]);
    length = getSegmentLength(segment);

    // Normalize the direction vector
    direction /= length;

    double slope = tan(getSegmentAngle(segment)*CV_PI/180);
    double intercept =  segment[1] - slope * segment[0];

    cv::Point2f start;
    if(slope > 0) {
        start = leftEndpoint + cv::Point2f(direction[0] * length * POSITIVE_SLOPE_SCALE, direction[1] * length * POSITIVE_SLOPE_SCALE);
    }
    else {
        start = leftEndpoint + cv::Point2f(direction[0] * length * NEGATIVE_SLOPE_SCALE, direction[1] * length * NEGATIVE_SLOPE_SCALE);
    }

    // Find the perpendicular direction (rotate by 90 degrees)
    cv::Vec2f perpendicularDirection(-direction[1], direction[0]);
    double searchLength = std::min(SEARCH_LENGTH_SCALE * length, MAX_SEARCH_LENGTH);

    // Create the perpendicular segment from the rightmost endpoint
    cv::Point2f perpendicularFromRightmostEndpoint = start + cv::Point2f(perpendicularDirection[0] * searchLength, perpendicularDirection[1] * searchLength);

    // Initialize variables to store the closest intersection point
    cv::Point2f closestIntersection;
    double minDistance = std::numeric_limits<double>::max();
    bool foundIntersection = false;
    cv::Vec4f closestSegment;

    // Check intersection of the perpendicular segment with every other segment
    for (const cv::Vec4f& otherSegment : segments) {
        cv::Point2f intersection;
        cv::Vec4f perpendicularVector = cv::Vec4f(start.x, start.y, perpendicularFromRightmostEndpoint.x, perpendicularFromRightmostEndpoint.y);
        cv::Vec4f extendedSegment1 = extendSegment(otherSegment, EXTENSION_SCALE);
        if (otherSegment != segment && segmentsIntersect(extendedSegment1, perpendicularVector, intersection)) {

            double dist = cv::norm(start - intersection);
            // last conditions to ensure that close segments of another parking slot line does not interfere
            if (dist > LOWER_BOUND_DISTANCE && dist < minDistance) { 
                minDistance = dist;
                closestIntersection = intersection;
                foundIntersection = true;
                closestSegment = otherSegment;           
            }
        }
    }

    // If an intersection is found, use it to build the rotated rect
    if (foundIntersection) {
        // Get the two endpoints of the second segment
        cv::Point2f endpoint1(closestSegment[0], closestSegment[1]);
        cv::Point2f endpoint2(closestSegment[2], closestSegment[3]);

        cv::RotatedRect boundingBox;
        if(slope>0) {
            cv::Point2f destinationRight(endpoint2.x, endpoint2.x * slope + intercept);
            cv::Point2f destinationBottom = destinationRight + cv::Point2f(perpendicularDirection[0] * minDistance, perpendicularDirection[1] * minDistance);
            boundingBox = cv::RotatedRect(leftEndpoint, destinationRight, destinationBottom);        
            } 
        else {
            cv::Point2f destinationLeft((endpoint1.y - intercept) / slope, endpoint1.y);
            cv::Point2f destinationUp = destinationLeft + cv::Point2f(perpendicularDirection[0] * minDistance, perpendicularDirection[1] * minDistance);
            boundingBox = cv::RotatedRect(rightEndpoint, destinationLeft, destinationUp);
        }
        
        cv::Point2f vertices[4];
        boundingBox.points(vertices);

        boundingBox.center = cv::Point2f(boundingBox.center.x + CENTER_SHIFT, boundingBox.center.y - CENTER_SHIFT);

        return boundingBox;
    }

    // If no intersection is found, return a default (empty) rotated rect
    return cv::RotatedRect();
}
#include "parkingSpotDetector.hpp"


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

void detectParkingSpotInImage(const cv::Mat& image, std::vector<ParkingSpot>& parkingSpots) {
	std::vector<cv::RotatedRect> spots;
    cv::Mat preprocessed = preprocessFindWhiteLines(image);
    cv::Mat intermediate_results = image.clone();
    
	cv::Ptr<cv::LineSegmentDetector > lsd = cv::createLineSegmentDetector();
    std::vector<cv::Vec4f> line_segm;
    lsd->detect(preprocessed,line_segm);
    std::vector<cv::Vec4f> segments;

	//Reject short lines
    for(int i = 0; i<line_segm.size(); i++) {                                                                                                                         //150
        if(getSegmentLength(line_segm[i]) > 35) {
            segments.push_back(line_segm[i]);
        }
    }
    double distance_threshold;
    std::vector<cv::Vec4f> filtered_segments = filterSegmentsNearTopRight(segments, cv::Size(image.cols,image.rows));

    std::vector<double> pos_angles;
    std::vector<double> neg_angles;
    std::vector<double> pos_lengths;
    std::vector<double> neg_lengths;

    for (const cv::Vec4f& segment : filtered_segments) {
        // Calculate the angle with respect to the x-axis
        double angle = getSegmentAngle(segment);

        // Calculate the length of the segment (line width)
        double length = getSegmentLength(segment);

        // Categorize and store the angle and length
        if (angle > 0) {
            pos_angles.push_back(angle);
            pos_lengths.push_back(length);

        } else if (angle < 0) {
            neg_angles.push_back(angle);
            neg_lengths.push_back(length);
        }
    }

    double avg_pos_angle = computeAvg(pos_angles);
    double avg_neg_angle = computeAvg(neg_angles);
    double avg_pos_width = computeAvg(pos_lengths);
    double avg_neg_width = computeAvg(neg_lengths);

    preprocessed = preprocessFindParkingLines(image);
 
    // offsets from avg values
    std::vector<int> angle_offsets = {-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6};
    std::vector<float> length_scales = {1.25};
    std::vector<cv::RotatedRect> list_boxes;
    std::vector<float> rect_scores(list_boxes.size(), -1); // Initialize scores with -1 for non-existing rects

    for(int l = 0; l<length_scales.size(); l++) {
        for(int k = 0; k<angle_offsets.size(); k++) {
            int template_width = avg_pos_width*length_scales[l];
            int template_height = 4;
            double angle = -avg_pos_angle+angle_offsets[k]; // negative

            std::vector<cv::Mat> rotated_template_and_mask = generateTemplate(template_width, angle, false);
            cv::Mat rotated_template = rotated_template_and_mask[0];
            cv::Mat rotated_mask = rotated_template_and_mask[1];
            cv::Mat tm_result_unnorm;
            cv::Mat tm_result;
            cv::matchTemplate(preprocessed,rotated_template,tm_result_unnorm,cv::TM_SQDIFF,rotated_mask);
            cv::normalize( tm_result_unnorm, tm_result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );
            // Finding local minima
            cv::Mat eroded;
            std::vector<cv::Point> minima;
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(rotated_template.cols, rotated_template.rows));
            cv::erode(tm_result, eroded, kernel);
            cv::Mat localMinimaMask = (tm_result == eroded) & (tm_result <= 0.2);

            // Find all non-zero points (local minima) in the mask
            cv::findNonZero(localMinimaMask, minima);

            // Iterate through each local minimum and process them
            for (const cv::Point& pt : minima) {
                // Calculate score based on the value in tm_result_unnorm at pt
                float score = tm_result_unnorm.at<float>(pt);

                // Get center of the bbox to draw the rotated rect
                cv::Point center;
                center.x = pt.x + rotated_template.cols / 2;
                center.y = pt.y + rotated_template.rows / 2;
                    //*1.25 = boundingbox scale
                cv::RotatedRect rotated_rect(center, cv::Size(template_width*1.25, template_height*1.25), -angle);
                
                // Check overlap with existing rects in list_boxes2
                bool overlaps = false;
                std::vector<size_t> overlapping_indices;

                for (size_t i = 0; i < list_boxes.size(); ++i) {
                    if (areRectsOverlapping(rotated_rect, list_boxes[i]) && rotated_rect.size.area() == list_boxes[i].size.area()) {
                        overlaps = true;
                        overlapping_indices.push_back(i);
                    }
                }

                // Determine whether to add the current rect
                if (!overlaps) {
                    // No overlap, add the rect directly
                    list_boxes.push_back(rotated_rect);
                    rect_scores.push_back(score);
                } 
                else {
                    // Handle overlap case: check if the current rect's score is higher (lower is good because we use sqdiff)
                    bool add_current_rect = true;
                    for (size_t idx : overlapping_indices) {
                        if (rotated_rect.size.area() < list_boxes[idx].size.area() || (rotated_rect.size.area() == list_boxes[idx].size.area() && score >= rect_scores[idx])) {
                            // The current rect has a higher or equal score, so don't add it
                            add_current_rect = false;
                            break;
                        }
                    }

                    if (add_current_rect) {
                        // Replace overlapping rects with the current one
                        for (size_t idx : overlapping_indices) {
                            list_boxes[idx] = rotated_rect;  // Replace the rect
                            rect_scores[idx] = score;         // Update the score
                        }
                    }
                }
            }
        }
    }
  

    std::vector<cv::RotatedRect> list_boxes2;
    std::vector<float> rect_scores2(list_boxes2.size(), -1); // Initialize scores with -1 for non-existing rects

    angle_offsets = {-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    length_scales = {1.1};
    for(int l = 0; l<length_scales.size(); l++) {
        for(int k = 0; k<angle_offsets.size(); k++) {
            int template_width = avg_neg_width*length_scales[l];
            int template_height = 4;
            double angle = avg_neg_angle+angle_offsets[k]; // negative

            std::vector<cv::Mat> rotated_template_and_mask = generateTemplate(template_width, angle, true);
            cv::Mat rotated_template = rotated_template_and_mask[0];
            cv::Mat rotated_mask = rotated_template_and_mask[1];
                            
            cv::Mat tm_result_unnorm;
            cv::Mat tm_result;
            cv::matchTemplate(preprocessed,rotated_template,tm_result_unnorm,cv::TM_SQDIFF,rotated_mask);
            cv::normalize( tm_result_unnorm, tm_result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );

             // Finding local minima
            cv::Mat eroded;
            std::vector<cv::Point> minima;
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(rotated_template.cols, rotated_template.rows));
            cv::erode(tm_result, eroded, kernel);
            cv::Mat local_minima_mask = (tm_result == eroded) & (tm_result <= 0.2);

            // Find all non-zero points (local minima) in the mask
            cv::findNonZero(local_minima_mask, minima);

            // Iterate through each local minimum and process them
            for (const cv::Point& pt : minima) {
                // Calculate score based on the value in tm_result_unnorm at pt
                float score = tm_result_unnorm.at<float>(pt);

                // Get center of the bbox to draw the rotated rect
                cv::Point center;
                center.x = pt.x + rotated_template.cols / 2;
                center.y = pt.y + rotated_template.rows / 2;

                // passare come parametro uno scale di questo rotatedrect perchè in quelli orizzontali serve fare*1.25width e height
                cv::RotatedRect rotated_rect(center, cv::Size(template_width, template_height), angle);
                cv::Point2f vertices[4];
                rotated_rect.points(vertices);
                for (int i = 0; i < 4; i++) {
                    //cv::line(image, vertices[i], vertices[(i+1) % 4], cv::Scalar(0, 255, 0), 2);
                }

                // Check overlap with existing rects in list_boxes2
                bool overlaps = false;
                std::vector<size_t> overlapping_indices;

                for (size_t i = 0; i < list_boxes2.size(); ++i) {
                    if (areRectsOverlapping(rotated_rect, list_boxes2[i]) && rotated_rect.size.area() == list_boxes[i].size.area()) {
                        overlaps = true;
                        overlapping_indices.push_back(i);
                    }
                }

                // Determine whether to add the current rect
                if (!overlaps) {
                    // No overlap, add the rect directly
                    list_boxes2.push_back(rotated_rect);
                    rect_scores2.push_back(score);
                } 
                else {
                    // Handle overlap case: check if the current rect's score is higher
                    bool add_current_rect = true;
                    for (size_t idx : overlapping_indices) {
                        if (rotated_rect.size.area() < list_boxes2[idx].size.area() || (rotated_rect.size.area() == list_boxes2[idx].size.area() && score >= rect_scores2[idx])) {
                            // The current rect has a lower or equal score, so don't add it
                            add_current_rect = false;
                            break;
                        }
                    }

                    if (add_current_rect) {
                        // Replace overlapping rects with the current one
                        for (size_t idx : overlapping_indices) {
                            list_boxes2[idx] = rotated_rect;  // Replace the rect
                            rect_scores2[idx] = score;         // Update the score
                        }
                    }
                }
            }
        }
    }

    std::vector<cv::RotatedRect> merged_pos_rects = mergeOverlappingRects(list_boxes);
    std::vector<cv::RotatedRect> merged_neg_rects = mergeOverlappingRects(list_boxes2);

    std::vector<cv::Vec4f> segments_pos;
    std::vector<cv::Vec4f> segments_neg;

    // Loop through all bounding boxes
    for (const cv::RotatedRect& rect : merged_pos_rects) {
        cv::Vec4f line_segment = convertRectToLine(rect);
        segments_pos.push_back(line_segment);
    }

    // Loop through all bounding boxes
    for (const cv::RotatedRect& rect : merged_neg_rects) {
        cv::Vec4f line_segment = convertRectToLine(rect);
        segments_neg.push_back(line_segment);
    }

    std::vector<cv::Vec4f> no_top_right_neg = filterSegmentsNearTopRight(segments_neg,cv::Size(image.cols,image.rows));
    std::vector<cv::Vec4f> no_top_right_pos = filterSegmentsNearTopRight(segments_pos,cv::Size(image.cols,image.rows));

    distance_threshold = avg_pos_width*0.4;
    std::vector<cv::Vec4f> filtered_segments_neg = filterCloseSegments(no_top_right_neg, distance_threshold);
    distance_threshold = avg_neg_width*0.4;
    std::vector<cv::Vec4f> filtered_segments_pos = filterCloseSegments(no_top_right_pos, distance_threshold);

    std::vector<cv::Vec4f> trimmed_segments_neg;
    for(cv::Vec4f& line_neg: filtered_segments_neg) {
        double length = getSegmentLength(line_neg);
        for(cv::Vec4f line_pos: filtered_segments_pos) {
            trimIfIntersect(line_neg,line_pos);
            double length_trimmed = getSegmentLength(line_neg);
        }        
        // cv::line(image, cv::Point(line_neg[0], line_neg[1]), cv::Point(line_neg[2], line_neg[3]), 
        // cv::Scalar(255, 0, 0), 2, cv::LINE_AA); 
    }

  // Thresholds
    float angle_threshold = CV_PI / 180.0f * 3;  // 10 degrees
    float length_threshold = 3;  // 30 pixels


    // Process the segments
    std::vector<cv::RotatedRect> rotated_rects = buildRotateRectsFromSegments(filtered_segments_pos);
    std::vector<cv::RotatedRect> rotated_rects2 = buildRotateRectsFromSegments(filtered_segments_neg);

    // Apply NMS filtering
    std::vector<cv::RotatedRect> elementsToRemove;
    nonMaximumSuppression(rotated_rects, elementsToRemove,0.3,false);
    std::vector<cv::RotatedRect> elementsToRemove2;
    nonMaximumSuppression(rotated_rects2, elementsToRemove2, 0.15,false);

    // Remove the elements determined by NMS filtering
    for (cv::RotatedRect element : elementsToRemove) {
        std::vector<cv::RotatedRect>::const_iterator iterator = elementIterator(rotated_rects, element);
        if (iterator != rotated_rects.cend()) {
            rotated_rects.erase(iterator);
        }
    }

    // Remove the elements determined by NMS filtering
    for (cv::RotatedRect element : elementsToRemove2) {
        std::vector<cv::RotatedRect>::const_iterator iterator = elementIterator(rotated_rects2, element);
        if (iterator != rotated_rects2.cend()) {
            rotated_rects2.erase(iterator);
        }
    }

    std::vector filtered_rects = filterBySurrounding(rotated_rects2, rotated_rects, image);
    std::vector<double> areas;
    double median_area;

    for (const auto& rect : rotated_rects) {
        if(rect.size.area()<1) continue;
        areas.push_back(rect.size.area());
    }
    median_area = computeMedian(areas);
    areas.clear();

    std::vector<cv::RotatedRect> remove_big_small_pos;
    std::vector<cv::RotatedRect> allRectsFound;

    for (const auto& rect : rotated_rects) {
        if(rect.size.area()>median_area/4 && rect.size.area()<median_area*2.25) {
            remove_big_small_pos.push_back(rect);
        }
    }

    for (const auto& rect : filtered_rects) {
        if(rect.size.area()<1) continue;
        areas.push_back(rect.size.area());
    }

    std::vector<cv::RotatedRect> remove_big_small_2;
    median_area = computeMedian(areas);

    for (const auto& rect : filtered_rects) {
        if(rect.size.area()>median_area/4 && rect.size.area()<median_area*2.25) {
            remove_big_small_2.push_back(rect);
            allRectsFound.push_back(rect);
        }
    }

    // Set the amount to shift when resolving overlaps
    float shift_amount = 5.0;

    // Resolve overlaps between vector1 and vector2
    resolve_overlaps(remove_big_small_2, remove_big_small_pos, shift_amount);

    // re-do nms after overlaps are resolved
    std::vector<cv::RotatedRect> elementsToRemove3;
    nonMaximumSuppression(remove_big_small_pos, elementsToRemove3, 0.5,false);

    // Remove the elements determined by NMS filtering
    for (cv::RotatedRect element : elementsToRemove3) {
        std::vector<cv::RotatedRect>::const_iterator iterator = elementIterator(remove_big_small_pos, element);
        if (iterator != remove_big_small_pos.cend()) {
            remove_big_small_pos.erase(iterator);
        }
    }

    for (const auto& rect : remove_big_small_pos) {
        if(rect.size.area()>1) { // the if is needed because removing with the iterator produces rects with zero area
            allRectsFound.push_back(rect);
        }
    }

    std::vector<cv::RotatedRect> all_close_rects;
    for(const cv::RotatedRect& rect:allRectsFound) {
        if(!isAlone(rect,allRectsFound)) {
            all_close_rects.push_back(rect);
            cv::Point2f vertices[4];
            rect.points(vertices);
        }
    }

	for (const cv::RotatedRect& rect : all_close_rects) {
        parkingSpots.push_back(ParkingSpot(0, 1, false, rect));
    }
}

/**
 * @brief Constructs rotated rectangles from a vector of segments.
 * 
 * This function processes a vector of segments to build rotated rectangles. For each segment in the input vector:
 * 1. It calls the `buildRotateRectFromPerpendicular` function to create a rotated rectangle based on the segment.
 * 2. It checks if the resulting rotated rectangle has a size area greater than a predefined minimum area.
 * 3. Only rectangles meeting the area criterion are added to the result vector.
 * 
 * @param segments A vector of `cv::Vec4f` representing the segments, where each `cv::Vec4f` contains coordinates of a segment (x1, y1, x2, y2).
 * @return A vector of `cv::RotatedRect` representing the constructed rotated rectangles that meet the size area requirement.
 */
std::vector<cv::RotatedRect> buildRotateRectsFromSegments(const std::vector<cv::Vec4f>& segments) {
    std::vector<cv::RotatedRect> rotatedRects;

    // Iterate over each segment
    for (const cv::Vec4f& segment : segments) {
        // Process each segment and build the rotated rectangle
        cv::RotatedRect rotatedRect = buildRotateRectFromPerpendicular(segment, segments);
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
cv::RotatedRect buildRotateRectFromPerpendicular(const cv::Vec4f& segment, const std::vector<cv::Vec4f>& segments) {
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
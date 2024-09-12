#include "parkingSpotDetector.hpp"

/*
TODO: 
Per ogni linea di segments_pos andare avanti (o indietro) finchè non si tocca una linea di segment_neg
Se poi le cose si overlappano, si tolgono le intersezioni o si uniscono
check if, by extending them up to a certain thold, they intersect
*/

// Function to detect parking spots in the images
void detectParkingSpots(const std::vector<cv::Mat>& images, std::vector<ParkingSpot>& parkingSpots) {
    
    std::vector<std::vector<ParkingSpot>> parkingSpotsPerImage;
    for(const auto& image : images) {
        // Find parking spots for each image separately
        parkingSpotsPerImage.push_back(detectParkingSpotInImage(image));
    }
}

// This function detects the parking spots in a single image
std::vector<ParkingSpot> detectParkingSpotInImage(const cv::Mat& image) {
    std::vector<ParkingSpot> parkingSpots;
	std::vector<cv::RotatedRect> spots;
    cv::Mat preprocessed = preprocess_find_white_lines(image);
    cv::Mat intermediate_results = image.clone();
    
	cv::Ptr<cv::LineSegmentDetector > lsd = cv::createLineSegmentDetector();
    std::vector<cv::Vec4f> line_segm;
    lsd->detect(preprocessed,line_segm);
    std::vector<cv::Vec4f> segments;

	//Reject short lines
    for(int i = 0; i<line_segm.size(); i++) {                                                                                                                         //150
        if(get_segment_length(line_segm[i]) > 35) {
            segments.push_back(line_segm[i]);
        }
    }
    double distance_threshold = 200.0;
    std::vector<cv::Vec4f> filtered_segments = filter_segments_near_top_right(segments, cv::Size(image.cols,image.rows), distance_threshold);

    std::vector<double> pos_angles;
    std::vector<double> neg_angles;
    std::vector<double> pos_lengths;
    std::vector<double> neg_lengths;

    for (const cv::Vec4f& segment : filtered_segments) {
        // Calculate the angle with respect to the x-axis
        double angle = get_segment_angular_coefficient(segment);

        // Calculate the length of the segment (line width)
        double length = get_segment_length(segment);

        // Categorize and store the angle and length
        if (angle > 0) {
            pos_angles.push_back(angle);
            pos_lengths.push_back(length);

            cv::line(intermediate_results, cv::Point(segment[0], segment[1]), cv::Point(segment[2], segment[3]), 
                 cv::Scalar(0, 255, 0), 2, cv::LINE_AA); 
        } else if (angle < 0) {
            neg_angles.push_back(angle);
            neg_lengths.push_back(length);

            cv::line(intermediate_results, cv::Point(segment[0], segment[1]), cv::Point(segment[2], segment[3]), 
            cv::Scalar(255, 0, 0), 2, cv::LINE_AA); 
        }
    }

    double avg_pos_angle = compute_avg(pos_angles);
    double avg_neg_angle = compute_avg(neg_angles);
    double avg_pos_width = compute_avg(pos_lengths);
    double avg_neg_width = compute_avg(neg_lengths);

    std::cout << "Median positive angle: " << avg_pos_angle << " degrees" << std::endl;
    std::cout << "Median negative angle: " << avg_neg_angle << " degrees" << std::endl;
    std::cout << "Median width of positive angle lines: " << avg_pos_width << std::endl;
    std::cout << "Median width of negative angle lines: " << avg_neg_width << std::endl;

    // Display the result
    cv::imshow("Detected Line Segments", intermediate_results);
    cv::waitKey(0);

    preprocessed = preprocess_find_parking_lines(image);
    cv::imshow("TM Input", preprocessed);
    cv::waitKey(0);

    // offsets from avg values
    std::vector<int> angle_offsets = {-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8};
    std::vector<float> length_scales = {1,1.5,2};
    std::vector<cv::RotatedRect> list_boxes;
    std::vector<float> rect_scores(list_boxes.size(), -1); // Initialize scores with -1 for non-existing rects

    for(int l = 0; l<length_scales.size(); l++) {
        for(int k = 0; k<angle_offsets.size(); k++) {
            int template_width = avg_pos_width*length_scales[l];
            int template_height = 4;
            double angle = -avg_pos_angle+angle_offsets[k]; // negative

            std::vector<cv::Mat> rotated_template_and_mask = generate_template(template_width, template_height, angle, false);
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
            cv::Mat localMinimaMask = (tm_result == eroded) & (tm_result <= 0.4);

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

                cv::RotatedRect rotated_rect(center, cv::Size(template_width*1, template_height*1), -angle);

                // Check overlap with existing rects in list_boxes2
                bool overlaps = false;
                std::vector<size_t> overlapping_indices;

                for (size_t i = 0; i < list_boxes.size(); ++i) {
                    if (are_rects_overlapping(rotated_rect, list_boxes[i]) && rotated_rect.size.area() == list_boxes[i].size.area()) {
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

    for(int l = 0; l<length_scales.size(); l++) {
        for(int k = 0; k<angle_offsets.size(); k++) {
            int template_width = avg_neg_width*length_scales[l];
            int template_height = 4;
            double angle = avg_neg_angle+angle_offsets[k]; // negative

            std::vector<cv::Mat> rotated_template_and_mask = generate_template(template_width, template_height, angle, true);
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
            cv::Mat local_minima_mask = (tm_result == eroded) & (tm_result <= 0.4);

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

                cv::RotatedRect rotated_rect(center, cv::Size(template_width*1, template_height*1), angle);

                // Check overlap with existing rects in list_boxes2
                bool overlaps = false;
                std::vector<size_t> overlapping_indices;

                for (size_t i = 0; i < list_boxes2.size(); ++i) {
                    if (are_rects_overlapping(rotated_rect, list_boxes2[i]) && rotated_rect.size.area() == list_boxes[i].size.area()) {
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

    std::vector<cv::RotatedRect> merged_pos_rects = merge_overlapping_rects(list_boxes);
    std::vector<cv::RotatedRect> merged_neg_rects = merge_overlapping_rects(list_boxes2);

    // std::vector<cv::RotatedRect> merged_pos_rects = merge_overlapping_rects(list_boxes);
    // std::vector<cv::RotatedRect> merged_neg_rects = merge_overlapping_rects(list_boxes2);
    std::vector<cv::Vec4f> segments_pos;
    std::vector<cv::Vec4f> segments_neg;

    // Loop through all bounding boxes
    for (const auto& rect : merged_pos_rects) {
        cv::Vec4f line_segment = convert_rect_to_line(rect);
        segments_pos.push_back(line_segment);
    }

    // Loop through all bounding boxes
    for (const auto& rect : merged_neg_rects) {
        cv::Vec4f line_segment = convert_rect_to_line(rect);
        segments_neg.push_back(line_segment);
    }

    distance_threshold = avg_pos_width*0.4;
    std::vector<cv::Vec4f> filtered_segments_neg = filter_close_segments(segments_neg, distance_threshold);
    distance_threshold = avg_neg_width*0.4;
    std::vector<cv::Vec4f> filtered_segments_pos = filter_close_segments(segments_pos, distance_threshold);

  // Thresholds
    float angle_threshold = CV_PI / 180.0f * 3;  // 10 degrees
    float length_threshold = 3;  // 30 pixels

    // Merge parallel and nearby segments
    std::vector<cv::Vec4f> merged_segments = merge_parallel_segments(filtered_segments_pos,angle_threshold,length_threshold);

    for (const auto& line_segment : filtered_segments_pos) {
        // Draw the line on the image
        cv::line(image, cv::Point2f(line_segment[0], line_segment[1]),
                 cv::Point2f(line_segment[2], line_segment[3]), cv::Scalar(0, 0, 255), 2);
    }

    // Loop through all bounding boxes
    for (const auto& line_segment : filtered_segments_neg) {
        // Draw the line on the image
        cv::line(image, cv::Point2f(line_segment[0], line_segment[1]),
                 cv::Point2f(line_segment[2], line_segment[3]), cv::Scalar(255, 0, 0), 2);
    }

    std::cout << "number of segments " << filtered_segments_pos.size() << std::endl;

    // Process the segments
    std::vector<cv::RotatedRect> rotated_rects = process_segments(filtered_segments_pos, image);

    // Output the result
    for (const auto& rect : rotated_rects) {
        cv::Point2f vertices[4];
        rect.points(vertices);
        for (int i = 0; i < 4; i++) {
            cv::line(image, vertices[i], vertices[(i+1) % 4], cv::Scalar(0, 255, 0), 2);
        }
    }

    cv::imshow("Intersection of Rotated Rects", image);
    cv::waitKey(0);

	return parkingSpots;
}

// Function to check overlap between two rotated rectangles
bool are_rects_overlapping(const cv::RotatedRect& rect1, const cv::RotatedRect& rect2) {
    std::vector<cv::Point2f> intersection_points;
    int intersection_status = cv::rotatedRectangleIntersection(rect1, rect2, intersection_points);
    return intersection_status == cv::INTERSECT_FULL || intersection_status == cv::INTERSECT_PARTIAL;
}

double compute_avg(std::vector<double>& data) {
    if (data.empty()) return 0.0;
    auto const count = static_cast<float>(data.size());
    return std::reduce(data.begin(), data.end()) / count;
}

float get_segment_angular_coefficient(const cv::Vec4f& segment) {
    float x1 = segment[0];
    float y1 = segment[1];
    float x2 = segment[2];
    float y2 = segment[3];

    return std::atan((y2 - y1) / (x2 - x1))*180/CV_PI;
}

float get_segment_length(const cv::Vec4f& segment) {
    float x1 = segment[0];
    float y1 = segment[1];
    float x2 = segment[2];
    float y2 = segment[3];

    return std::sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

// Helper function to check if two rotated rectangles are aligned (same angle within a tolerance)
bool are_rects_aligned(const cv::RotatedRect& rect1, const cv::RotatedRect& rect2, float angle_tolerance) {
    return std::abs(rect1.angle - rect2.angle) <= angle_tolerance;
}

// Function to get the right-most endpoint of a segment
cv::Point2f get_rightmost_endpoint(const cv::Vec4f& segment) {
    cv::Point2f p1(segment[0], segment[1]);  // First endpoint (x1, y1)
    cv::Point2f p2(segment[2], segment[3]);  // Second endpoint (x2, y2)

    // Compare the x-coordinates to find the right-most point
    return (p1.x > p2.x) ? p1 : p2;
}

// Function to check if two segments intersect
bool segments_intersect(const cv::Vec4f& seg1, const cv::Vec4f& seg2, cv::Point2f& intersection) {
    cv::Point2f p1(seg1[0], seg1[1]), q1(seg1[2], seg1[3]);
    cv::Point2f p2(seg2[0], seg2[1]), q2(seg2[2], seg2[3]);

    cv::Vec2f r = cv::Vec2f(q1 - p1);  // Direction of seg1
    cv::Vec2f s = cv::Vec2f(q2 - p2);  // Direction of seg2

    float rxs = r[0] * s[1] - r[1] * s[0];
    cv::Vec2f qp = cv::Vec2f(p2 - p1);
    float t = (qp[0] * s[1] - qp[1] * s[0]) / rxs;
    float u = (qp[0] * r[1] - qp[1] * r[0]) / rxs;

    if (rxs != 0 && t >= 0 && t <= 1 && u >= 0 && u <= 1) {
        intersection = p1 + cv::Point2f(r[0] * t, r[1] * t);  // Convert r * t to cv::Point2f
        return true;
    }
    return false;
}

// Function to move along the perpendicular direction from the right-most endpoint
cv::RotatedRect build_rotatedrect_from_movement(const cv::Vec4f& segment, const std::vector<cv::Vec4f>& segments, cv::Mat image) {
    // Rightmost endpoint of the original segment
    cv::Point2f right_endpoint(segment[2], segment[3]);
    cv::Point2f left_endpoint(segment[0], segment[1]);

    // Compute the direction vector of the original segment
    cv::Vec2f direction = cv::Vec2f(right_endpoint - left_endpoint);
    float length = std::sqrt(direction[0] * direction[0] + direction[1] * direction[1]);

    // Normalize the direction
    direction /= length;

    // Find the perpendicular direction (rotate by 90 degrees)
    cv::Vec2f perpendicular_direction(-direction[1], direction[0]);

    // Create the perpendicular segment from the rightmost endpoint
    cv::Point2f perp_end = right_endpoint + cv::Point2f(perpendicular_direction[0] * length, perpendicular_direction[1] * length);

    // Initialize variables to store the closest intersection point
    cv::Point2f closest_intersection;
    float min_distance = std::numeric_limits<float>::max();
    bool found_intersection = false;
    cv::Vec4f per_vect = cv::Vec4f(right_endpoint.x, right_endpoint.y, perp_end.x, perp_end.y);

    // Check intersection of the perpendicular segment with every other segment
    for (const auto& other_segment : segments) {
        cv::Point2f intersection;
        if (other_segment != segment && segments_intersect(cv::Vec4f(right_endpoint.x, right_endpoint.y, perp_end.x, perp_end.y), other_segment, intersection)) {
            float dist = cv::norm(right_endpoint-intersection);
            if (dist < min_distance && dist < get_segment_length(segment)/2) {
                min_distance = dist;
                closest_intersection = intersection;
                found_intersection = true;
            }
        }
    }

    // If an intersection is found, use it to build the rotated rect
    if (found_intersection) {
        return cv::RotatedRect(left_endpoint, right_endpoint, closest_intersection);
    }

    // If no intersection is found, return a default (empty) rotated rect
    return cv::RotatedRect();
}

// Main function to process segments
std::vector<cv::RotatedRect> process_segments(const std::vector<cv::Vec4f>& segments, cv::Mat image) {
    std::vector<cv::RotatedRect> rotated_rects;

    // Iterate over each segment
    for (const auto& segment : segments) {
        // Process each segment and build the rotated rectangle
        cv::RotatedRect rotated_rect = build_rotatedrect_from_movement(segment, segments, image);
        rotated_rects.push_back(rotated_rect);
    }

    return rotated_rects;
}

// Function to merge overlapping and aligned rotated rectangles
std::vector<cv::RotatedRect> merge_overlapping_rects(std::vector<cv::RotatedRect>& rects) {
    std::vector<cv::RotatedRect> merged_rects;
    std::vector<bool> merged(rects.size(), false);  // Track which rects have been merged

    for (size_t i = 0; i < rects.size(); ++i) {
        if (merged[i]) continue;  // Skip already merged rects

        // Start a new group of merged rects
        std::vector<cv::Point2f> group_points;
        cv::Point2f points[4];
        rects[i].points(points);
        group_points.insert(group_points.end(), points, points + 4);
        merged[i] = true;

        // Check for overlap and alignment with other rects
        for (size_t j = i + 1; j < rects.size(); ++j) {
            if (!merged[j] && are_rects_overlapping(rects[i], rects[j]) && are_rects_aligned(rects[i], rects[j],10)) {
                // Merge the overlapping and aligned rect
                rects[j].points(points);
                group_points.insert(group_points.end(), points, points + 4);
                merged[j] = true;
            }
        }

        // Create a single bounding box from the merged group points
        cv::RotatedRect merged_rect = cv::minAreaRect(group_points);
        merged_rects.push_back(merged_rect);
    }

    return merged_rects;
}

// Function to filter segments that are too close to the top-right corner
std::vector<cv::Vec4f> filter_segments_near_top_right(const std::vector<cv::Vec4f>& segments, const cv::Size& image_size, double distance_threshold) {
    // Define the top-right corner of the image
    cv::Point2f top_right_corner(image_size.width - 1, 0);

    std::vector<cv::Vec4f> filtered_segments;

    for (const auto& segment : segments) {
        // Extract the endpoints of the segment
        cv::Point2f p1(segment[0], segment[1]);
        cv::Point2f p2(segment[2], segment[3]);

        // Calculate the distances of both endpoints from the top-right corner
        double dist_p1 =cv::norm(p1-top_right_corner);
        double dist_p2 = cv::norm(p2-top_right_corner);

        // Only add the segment to the filtered list if both endpoints are farther than the threshold from the top-right corner
        if (dist_p1 > distance_threshold && dist_p2 > distance_threshold) {
            filtered_segments.push_back(segment);
        }
    }

    return filtered_segments;
}

// Helper function to compute the midpoint of a line segment
cv::Point2f compute_midpoint(const cv::Vec4f& segment) {
    return cv::Point2f((segment[0] + segment[2]) / 2.0f, (segment[1] + segment[3]) / 2.0f);
}

// Helper function to compute the perpendicular direction of a segment
cv::Point2f compute_perpendicular_direction(const cv::Vec4f& segment) {
    float dx = segment[2] - segment[0];
    float dy = segment[3] - segment[1];
    return cv::Point2f(-dy, dx);
}

// Function to calculate the distance between the midpoints of two segments
float compute_distance_between_segments(const cv::Vec4f& seg1, const cv::Vec4f& seg2) {
    cv::Point2f mid1 = compute_midpoint(seg1);
    cv::Point2f mid2 = compute_midpoint(seg2);
    return cv::norm(mid1 - mid2);  // Euclidean distance between midpoints
}

// Function to filter segments based on a distance threshold
std::vector<cv::Vec4f> filter_close_segments(const std::vector<cv::Vec4f>& segments, float distance_threshold) {
    std::vector<cv::Vec4f> filtered_segments;
    std::vector<bool> discarded(segments.size(), false);  // To mark discarded segments

    for (size_t i = 0; i < segments.size(); ++i) {
        if (discarded[i]) continue;  // Skip already discarded segments

        const cv::Vec4f& current_segment = segments[i];
        filtered_segments.push_back(current_segment);  // Add current segment to result

        // Compare with remaining segments and discard close ones
        for (size_t j = i + 1; j < segments.size(); ++j) {
            if (!discarded[j]) {
                float distance = compute_distance_between_segments(current_segment, segments[j]);
                if (distance < distance_threshold) {
                    discarded[j] = true;  // Mark this segment as discarded
                }
            }
        }
    }

    return filtered_segments;
}

// Function to split a segment into two smaller segments
std::vector<cv::Vec4f> split_segment(const cv::Vec4f& segment) {
    std::vector<cv::Vec4f> split_segments;
    
    // Find midpoint
    cv::Point2f p1(segment[0], segment[1]);
    cv::Point2f p2(segment[2], segment[3]);
    cv::Point2f midpoint((p1.x + p2.x) / 2, (p1.y + p2.y) / 2);
    
    // Split into two segments
    split_segments.push_back(cv::Vec4f(p1.x, p1.y, midpoint.x, midpoint.y));
    split_segments.push_back(cv::Vec4f(midpoint.x, midpoint.y, p2.x, p2.y));
    
    return split_segments;
}

// Helper function to merge two segments into one
cv::Vec4f merge_segments(const cv::Vec4f& seg1, const cv::Vec4f& seg2) {
    // Find the endpoints of the combined segment
    cv::Point2f points[] = {
        cv::Point2f(seg1[0], seg1[1]),
        cv::Point2f(seg1[2], seg1[3]),
        cv::Point2f(seg2[0], seg2[1]),
        cv::Point2f(seg2[2], seg2[3])
    };

    // Find the minimum and maximum x and y values to determine the new segment's endpoints
    cv::Point2f min_pt = points[0];
    cv::Point2f max_pt = points[0];

    for (int i = 1; i < 4; ++i) {
        if (points[i].x < min_pt.x || (points[i].x == min_pt.x && points[i].y < min_pt.y)) {
            min_pt = points[i];
        }
        if (points[i].x > max_pt.x || (points[i].x == max_pt.x && points[i].y > max_pt.y)) {
            max_pt = points[i];
        }
    }

    // Return the new segment
    return cv::Vec4f(min_pt.x, min_pt.y, max_pt.x, max_pt.y);
}

// Function to merge overlapping or near-parallel segments
std::vector<cv::Vec4f> merge_parallel_segments(std::vector<cv::Vec4f>& segments, float angle_threshold, float distance_threshold) {
    std::vector<cv::Vec4f> merged_segments;
    std::vector<bool> merged(segments.size(), false);  // Keep track of merged segments

    for (size_t i = 0; i < segments.size(); ++i) {
        if (merged[i]) continue;  // Skip already merged segments

        cv::Vec4f current_segment = segments[i];
        merged[i] = true;  // Mark the current segment as merged

        float current_angle = get_segment_angular_coefficient(current_segment);  // Compute the angle of the current segment

        // Compare with remaining segments
        for (size_t j = i + 1; j < segments.size(); ++j) {
            if (merged[j]) continue;  // Skip already merged segments

            // Compute the angle of the other segment
            float other_angle = get_segment_angular_coefficient(segments[j]);
            float angle_diff = std::abs(current_angle - other_angle);

            // Ensure the angle difference is within the specified threshold
            if (angle_diff > angle_threshold) continue;

            // Check if the segments are close enough in terms of distance
            cv::Point2f seg1_start(current_segment[0], current_segment[1]);
            cv::Point2f seg1_end(current_segment[2], current_segment[3]);
            cv::Point2f seg2_start(segments[j][0], segments[j][1]);
            cv::Point2f seg2_end(segments[j][2], segments[j][3]);

            // Compute the distances between the endpoints
            float dist1 = cv::norm(seg1_start - seg2_start);
            float dist2 = cv::norm(seg1_start - seg2_end);
            float dist3 = cv::norm(seg1_end - seg2_start);
            float dist4 = cv::norm(seg1_end - seg2_end);

            // If any of the distances is below the threshold, merge the segments
            if (dist1 < distance_threshold || dist2 < distance_threshold || dist3 < distance_threshold || dist4 < distance_threshold) {
                current_segment = merge_segments(current_segment, segments[j]);
                merged[j] = true;  // Mark this segment as merged
            }
        }

        // Add the merged segment to the result
        merged_segments.push_back(current_segment);
    }


    return merged_segments;
}
cv::Mat preprocess_find_white_lines(const cv::Mat& src) {
    cv::Mat filteredImage;
    cv::bilateralFilter(src, filteredImage, -1, 40, 10);

    cv::Mat gs;
    cv::cvtColor(filteredImage, gs, cv::COLOR_BGR2GRAY);

    cv::Mat adpt;
    cv::adaptiveThreshold(gs,adpt,255, cv::ADAPTIVE_THRESH_MEAN_C ,cv::THRESH_BINARY, 9,-20);

    cv::Mat gr_x;
    cv::Sobel(adpt, gr_x, CV_8U, 1,0);

    cv::Mat gr_y;
    cv::Sobel(adpt, gr_y, CV_8U, 0,1);

    cv::Mat magnitude = gr_x + gr_y;

    cv::imshow("grayscale", magnitude);
    cv::waitKey(0);

    cv::Mat element = cv::getStructuringElement( 
                        cv::MORPH_CROSS, cv::Size(3,3)); 

    // dil 2 erode 1
    cv::dilate(magnitude,adpt,element,cv::Point(-1,-1),4);
    cv::erode(adpt,adpt,element,cv::Point(-1,-1),3);

    cv::imshow("grayscale", adpt);
    cv::waitKey(0);
    return adpt;
}

cv::Mat preprocess_find_parking_lines(const cv::Mat& src) {
    cv::Mat filteredImage;
    cv::bilateralFilter(src, filteredImage, -1, 40, 10);

    cv::Mat gs;
    cv::cvtColor(filteredImage, gs, cv::COLOR_BGR2GRAY);

    cv::Mat gx;
    cv::Sobel(gs, gx, CV_16S, 1,0);

    cv::Mat gy;
    cv::Sobel(gs, gy, CV_16S, 0,1);

    cv::Mat abs_grad_x;
    cv::Mat abs_grad_y;
    cv::convertScaleAbs(gx, abs_grad_x);
    cv::convertScaleAbs(gy, abs_grad_y);

    cv::Mat element = cv::getStructuringElement( 
                        cv::MORPH_CROSS, cv::Size(3,3)); 

    cv::Mat grad_magn;
    cv::Mat grad_magn_proc;
    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad_magn);
    cv::adaptiveThreshold(grad_magn,grad_magn_proc,255, cv::ADAPTIVE_THRESH_GAUSSIAN_C ,cv::THRESH_BINARY, 45,-40);
    cv::dilate(grad_magn_proc,grad_magn_proc,element,cv::Point(-1,-1),1);
    cv::erode(grad_magn_proc,grad_magn_proc,element,cv::Point(-1,-1),1);
    cv::imshow("adaptive thold grad magn",grad_magn_proc+grad_magn);
    return grad_magn_proc;
    cv::Mat adpt;
    cv::waitKey(0);

    cv::Mat edges;
    cv::Canny(grad_magn, edges,150, 400);

    // todo: if it overlaps woth a perpendicular one, do minarearect

    cv::imshow("dilated canny",edges);
    cv::waitKey(0);

    return edges;
}

// Function to convert a narrow RotatedRect into a line segment
cv::Vec4f convert_rect_to_line(const cv::RotatedRect& rect) {
    cv::Point2f points[4];
    rect.points(points);  // Get the four corner points of the RotatedRect

    // Calculate the length of the edges
    float length1 = cv::norm(points[0] - points[1]);
    float length2 = cv::norm(points[1] - points[2]);
    float length3 = cv::norm(points[2] - points[3]);
    float length4 = cv::norm(points[3] - points[0]);

    // The longest two opposite edges define the line
    float max_length1 = std::max(length1, length3);
    float max_length2 = std::max(length2, length4);

    // Midpoints of the longest edges
    cv::Point2f midpoint1, midpoint2;

    if (max_length1 < max_length2) {
        // Use points 0->1 and 2->3 (longest edge pair)
        midpoint1 = (points[0] + points[1]) * 0.5f;
        midpoint2 = (points[2] + points[3]) * 0.5f;
    } else {
        // Use points 1->2 and 3->0 (other longest edge pair)
        midpoint1 = (points[1] + points[2]) * 0.5f;
        midpoint2 = (points[3] + points[0]) * 0.5f;
    }

    // Return the line segment as a vector of 4 floats (x1, y1, x2, y2)
    return cv::Vec4f(midpoint1.x, midpoint1.y, midpoint2.x, midpoint2.y);
}


std::vector<cv::Mat> generate_template(double width, double height, double angle, bool flipped){
    // angle is negative
    // Template size
    int template_height;
    int template_width;
    cv::Point rotation_center;
    double rotation_angle;
    float rotated_width;
    float rotated_height;

    // Rotate the template
    if(!flipped) {
        template_height = height;
        template_width = width;
        rotation_angle = angle; // negative rotation_angle for not flipped (angle is negative)
        rotated_width = template_width*cos(-rotation_angle*CV_PI/180)+template_height; // needs positive angle
        rotated_height = template_width*sin(-rotation_angle*CV_PI/180)+template_height; // needs positive angle
    }

    if(flipped) {
        template_height = width;
        template_width = height;
        rotation_angle = -90-angle; // negative rotation_angle for flipped (angle is negative)
        rotated_width = template_height*cos(-angle*CV_PI/180)+template_width; // needs positive angle
        rotated_height = template_height*sin(-angle*CV_PI/180)+template_width; // needs positive angle
    }

    // Horizontal template and mask definition
    cv::Mat horizontal_template(template_height,template_width,CV_8U);
    cv::Mat horizontal_mask(template_height,template_width,CV_8U);

    // Build the template and mask
    for(int i = 0; i< horizontal_template.rows; i++) {
        for(int j = 0; j<horizontal_template.cols; j++) {
            horizontal_template.at<uchar>(i,j) = 255;
            horizontal_mask.at<uchar>(i,j) = 255;
        }
    }

    rotation_center.y = template_height-1;
    rotation_center.x = 0;

    cv::Mat R = cv::getRotationMatrix2D(rotation_center, rotation_angle,1);
    cv::Mat rotated_template;
    cv::Mat rotated_mask;
    
    cv::warpAffine(horizontal_template,rotated_template,R,cv::Size(rotated_width,rotated_height));
    cv::warpAffine(horizontal_mask,rotated_mask,R,cv::Size(rotated_width,rotated_height));

    return std::vector<cv::Mat>{rotated_template,rotated_mask};
}

cv::Mat applyGammaTransform(const cv::Mat& src, double gamma) {
    // Create a lookup table for faster processing
    cv::Mat lookupTable(1, 256, CV_8U);
    uchar* p = lookupTable.ptr();
    for (int i = 0; i < 256; ++i) {
        p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
    }

    cv::Mat dst;
    // Apply the lookup table to the source image
    cv::LUT(src, lookupTable, dst);

    return dst;
}

double computeIntersectionArea(const cv::RotatedRect& rect1, const cv::RotatedRect& rect2) {
    // Converti i RotatedRect in vettori di punti (poligoni)
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

    // Calcola l'intersezione tra i due poligoni
    std::vector<cv::Point2f> intersection;
    double intersectionArea = cv::intersectConvexConvex(points1, points2, intersection) / std::min(area1, area2);

    return intersectionArea;
}

std::vector<cv::RotatedRect>::const_iterator elementIterator(const std::vector<cv::RotatedRect>& vec, const cv::RotatedRect& elem){
    for (auto it = vec.cbegin(); it != vec.cend(); ++it) {
        if (it->center.x == elem.center.x &&
            it->center.y == elem.center.y) 
        {
            return it; // Restituiamo l'iteratore all'elemento
        }
    }
    return vec.cend(); // Restituiamo end() se l'elemento non è stato trovato
}

void nms(std::vector<cv::RotatedRect> &vec, std::vector<cv::RotatedRect> &elementsToRemove) {
    for (const auto& rect1 : vec) {
        for (const auto& rect2 : vec) {
            if (!(rect1.center.x == rect2.center.x && rect1.center.y == rect2.center.y) && (computeIntersectionArea(rect1, rect2) > 0.75)) {
                if (rect1.size.area() > rect2.size.area()){
                    elementsToRemove.push_back(rect2);
                } else {
                    elementsToRemove.push_back(rect1);
                }
            }
        }
    }
}
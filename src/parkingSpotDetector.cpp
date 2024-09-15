#include "parkingSpotDetector.hpp"

/*
TODO: 
Per ogni linea di segments_pos andare avanti (o indietro) finchè non si tocca una linea di segment_neg
Se poi le cose si overlappano, si tolgono le intersezioni o si uniscono
check if, by extending them up to a certain thold, they intersect
for other lines: if there is another segment in between, split bounding box

if red ones are contained in green ones, then delete red one
else do something with intersection with green ones maybe


blues need to extend more like already done before
if a green overlaps with a red, it means that it should be more to the left, or where most of itself is
filter reds by area
filter reds if they are surrounded by greens
*/

// Function to detect parking spots in the images
void detectParkingSpots(const std::vector<cv::Mat>& images, std::vector<ParkingSpot>& parkingSpots) {
    
    // std::vector<cv::RotatedRect> parkingSpotsPerImage;
    std::vector<cv::RotatedRect> all_rects;
    cv::Mat final_result = images[0].clone();

    for(const auto& image : images) {
        // Find parking spots for each image separately
        for(cv::RotatedRect parking_spot_image : detectParkingSpotInImage(image)) {
            all_rects.push_back(parking_spot_image);
        }
    }

    std::vector<cv::RotatedRect> elements_to_remove;
    nms(all_rects, elements_to_remove, 0.3, true);

    // Remove the elements determined by NMS filtering
    for (cv::RotatedRect element : elements_to_remove) {
        std::vector<cv::RotatedRect>::const_iterator iterator = elementIterator(all_rects, element);
        if (iterator != all_rects.cend()) {
            all_rects.erase(iterator);
        }
    }

    std::vector<cv::Point2f> all_vertices;
    for (cv::RotatedRect& rect : all_rects) {
        if(rect.size.area()>1) { // the if is needed because removing with the iterator produces rects with zero area
            cv::Point2f vertices[4];
            rect.points(vertices);
            parkingSpots.push_back(ParkingSpot(rect));
            for (int i = 0; i < 4; i++) {
                all_vertices.push_back(vertices[i]);
                cv::line(final_result, vertices[i], vertices[(i+1) % 4], cv::Scalar(0, 255, 0), 2);
            }
        }
    }

    cv::imshow("Final detection", final_result);

    cv::imshow("minimap", build_minimap(parkingSpots));

    return;
}

// This function detects the parking spots in a single image
std::vector<cv::RotatedRect> detectParkingSpotInImage(const cv::Mat& image) {
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
    double distance_threshold;
    std::vector<cv::Vec4f> filtered_segments = filter_segments_near_top_right(segments, cv::Size(image.cols,image.rows));

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

            //cv::line(intermediate_results, cv::Point(segment[0], segment[1]), cv::Point(segment[2], segment[3]), 
            //     cv::Scalar(0, 255, 0), 2, cv::LINE_AA); 
        } else if (angle < 0) {
            neg_angles.push_back(angle);
            neg_lengths.push_back(length);

            //cv::line(intermediate_results, cv::Point(segment[0], segment[1]), cv::Point(segment[2], segment[3]), 
            //cv::Scalar(255, 0, 0), 2, cv::LINE_AA); 
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
    //cv::imshow("Detected Line Segments", intermediate_results);
    //cv::waitKey(0);

    preprocessed = preprocess_find_parking_lines(image);
    // cv::imshow("TM Input", preprocessed);
    //cv::waitKey(0);

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

                cv::RotatedRect rotated_rect(center, cv::Size(template_width*1.25, template_height*1.25), -angle);
                
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

    for(auto& rect:list_boxes) {
                cv::Point2f vertices[4];
        rect.points(vertices);
        for (int i = 0; i < 4; i++) {
            // cv::line(image, vertices[i], vertices[(i+1) % 4], cv::Scalar(0, 255, 0), 2);
        }
    }
    //cv::imshow("ppp", image);
    //cv::waitKey(0);

    std::vector<cv::RotatedRect> list_boxes2;
    std::vector<float> rect_scores2(list_boxes2.size(), -1); // Initialize scores with -1 for non-existing rects

    angle_offsets = {-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    length_scales = {1.1};
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

    // Output the result
    for (const auto& rect : list_boxes) {

    }

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

    std::vector<cv::Vec4f> no_top_right_neg = filter_segments_near_top_right(segments_neg,cv::Size(image.cols,image.rows));
    std::vector<cv::Vec4f> no_top_right_pos = filter_segments_near_top_right(segments_pos,cv::Size(image.cols,image.rows));

    distance_threshold = avg_pos_width*0.4;
    std::vector<cv::Vec4f> filtered_segments_neg = filter_close_segments(no_top_right_neg, distance_threshold);
    distance_threshold = avg_pos_width*0.4;
    std::vector<cv::Vec4f> filtered_segments_pos = filter_close_segments(no_top_right_pos, distance_threshold);

    std::vector<cv::Vec4f> trimmed_segments_neg;
    for(cv::Vec4f& line_neg: filtered_segments_neg) {
        double length = get_segment_length(line_neg);
        for(cv::Vec4f line_pos: filtered_segments_pos) {
            trim_if_intersect(line_neg,line_pos);
            double length_trimmed = get_segment_length(line_neg);
        }        
        // cv::line(image, cv::Point(line_neg[0], line_neg[1]), cv::Point(line_neg[2], line_neg[3]), 
        // cv::Scalar(255, 0, 0), 2, cv::LINE_AA); 
    }

  // Thresholds
    float angle_threshold = CV_PI / 180.0f * 3;  // 10 degrees
    float length_threshold = 3;  // 30 pixels
    // cv::imshow("ppp", image);
    // cv::waitKey(0);

    std::cout << "number of segments ORIGINAL" << segments_neg.size() << std::endl;
    std::cout << "number of segments FILTERED" << filtered_segments_neg.size() << std::endl;

    // Process the segments
    std::vector<cv::RotatedRect> rotated_rects = process_segments(filtered_segments_pos, image);
    std::vector<cv::RotatedRect> rotated_rects2 = process_segments(filtered_segments_neg, image);

    // Apply NMS filtering
    std::vector<cv::RotatedRect> elementsToRemove;
    nms(rotated_rects, elementsToRemove,0.3,false);
    std::vector<cv::RotatedRect> elementsToRemove2;
    nms(rotated_rects2, elementsToRemove2, 0.15,false);

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

    // Output the result
    std::vector<cv::Point2f> centers_all;

    // Output the result
    std::vector filtered_rects = filter_by_surrounding(rotated_rects2,rotated_rects,image);
    std::vector<double> areas;
    double median_area;

    for (const auto& rect : rotated_rects) {
        if(rect.size.area()<1) continue;
        areas.push_back(rect.size.area());
    }
    median_area = compute_median(areas);
    areas.clear();

    std::vector<cv::RotatedRect> remove_big_small_pos;
    std::vector<cv::RotatedRect> all_rects;
    std::cout << "median area " << median_area << std::endl;

    for (const auto& rect : rotated_rects) {
        std::cout << "rect area "<< rect.size.area() << std::endl;
        if(rect.size.area()>median_area/4 && rect.size.area()<median_area*2.25) {
            remove_big_small_pos.push_back(rect);
        }
    }

    for (const auto& rect : filtered_rects) {
        if(rect.size.area()<1) continue;
        areas.push_back(rect.size.area());
    }

    std::vector<cv::RotatedRect> remove_big_small_2;
    median_area = compute_median(areas);
    std::cout << "median area " << median_area << std::endl;

    for (const auto& rect : filtered_rects) {
        std::cout << "rect area "<< rect.size.area() << std::endl;
        if(rect.size.area()>median_area/4 && rect.size.area()<median_area*2.25) {
            remove_big_small_2.push_back(rect);
            all_rects.push_back(rect);
        }
    }

    // Set the amount to shift when resolving overlaps
    float shift_amount = 5.0;

    // Resolve overlaps between vector1 and vector2
    resolve_overlaps(remove_big_small_2, remove_big_small_pos, shift_amount);

    // re-do nms after overlaps are resolved
    std::vector<cv::RotatedRect> elementsToRemove3;
    nms(remove_big_small_pos, elementsToRemove3, 0.5,false);

    // Remove the elements determined by NMS filtering
    for (cv::RotatedRect element : elementsToRemove3) {
        std::vector<cv::RotatedRect>::const_iterator iterator = elementIterator(remove_big_small_pos, element);
        if (iterator != remove_big_small_pos.cend()) {
            remove_big_small_pos.erase(iterator);
        }
    }

    for (const auto& rect : remove_big_small_pos) {
        if(rect.size.area()>1) { // the if is needed because removing with the iterator produces rects with zero area
            cv::Point2f vertices[4];
            rect.points(vertices);
            for (int i = 0; i < 4; i++) {
                all_rects.push_back(rect);
            }
        }
    }

    std::vector<cv::RotatedRect> all_close_rects;
    for(const cv::RotatedRect& rect:all_rects) {
        if(!is_alone(rect,all_rects)) {
            centers_all.push_back(rect.center);
            all_close_rects.push_back(rect);
            cv::Point2f vertices[4];
            rect.points(vertices);
            for (int i = 0; i < 4; i++) {
                cv::line(image, vertices[i], vertices[(i+1) % 4], cv::Scalar(0, 255, 0), 2);
            }
        }
    }

    // cv::imshow("ppp", image);
    // cv::waitKey(0);

    // cv::imshow("result", minimap);

    //cv::imshow("hull image",image);
    //cv::waitKey(0);

	return all_close_rects;
}
void center_minimap(const cv::Mat& minimap, cv::Mat& large_image) {
    // Step 1: Ensure the large image is bigger than the minimap
    if (minimap.cols > large_image.cols || minimap.rows > large_image.rows) {
        std::cerr << "Error: The minimap is larger than the large image." << std::endl;
        return;
    }

    // Step 2: Calculate the top-left corner (offset) to center the minimap
    int offset_x = (large_image.cols - minimap.cols) / 2;
    int offset_y = (large_image.rows - minimap.rows) / 2;

    // Step 3: Create a Region of Interest (ROI) in the large image
    cv::Rect roi(offset_x, offset_y, minimap.cols, minimap.rows);
    cv::Mat large_image_roi = large_image(roi);

    // Step 4: Copy the minimap into the centered region of the large image
    minimap.copyTo(large_image_roi);
}

void align_rects(std::vector<cv::RotatedRect>& rects, float threshold) {
    // Sort the rects by their y-coordinate
    std::sort(rects.begin(), rects.end(), 
        [](const cv::RotatedRect& a, const cv::RotatedRect& b) {
            return a.center.y < b.center.y;
        });

    // Iterate through each unique y-coordinate and align rects within the threshold
    for (size_t i = 0; i < rects.size(); ++i) {
        float base_y = rects[i].center.y;

        // Align all rects within the threshold of the current base_y
        for (size_t j = i + 1; j < rects.size(); ++j) {
            if (std::fabs(rects[j].center.y - base_y) <= threshold) {
                rects[j].center.y = base_y;
            }
        }

        // Skip over rects already aligned to the current base_y
        while (i + 1 < rects.size() && std::fabs(rects[i + 1].center.y - base_y) <= threshold) {
            ++i;
        }
    }
}

// Function to find the corners (top-left, top-right, bottom-left, bottom-right) from 4 points
std::vector<cv::Point2f> find_corners(const std::vector<cv::Point2f>& points) {
    if (points.size() != 4) {
        throw std::invalid_argument("Input vector must contain exactly 4 points.");
    }

    std::vector<cv::Point2f> corners(4);

    // Sort points based on y-coordinate (top two first, bottom two last)
    std::vector<cv::Point2f> sorted_points = points;
    std::sort(sorted_points.begin(), sorted_points.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
        return a.y < b.y;
    });

    // Top-left and top-right points (first two in sorted list)
    if (sorted_points[0].x < sorted_points[1].x) {
        corners[0] = sorted_points[0];  // Top-left
        corners[1] = sorted_points[1];  // Top-right
    } else {
        corners[0] = sorted_points[1];  // Top-left
        corners[1] = sorted_points[0];  // Top-right
    }

    // Bottom-left and bottom-right points (last two in sorted list)
    if (sorted_points[2].x < sorted_points[3].x) {
        corners[2] = sorted_points[2];  // Bottom-left
        corners[3] = sorted_points[3];  // Bottom-right
    } else {
        corners[2] = sorted_points[3];  // Bottom-left
        corners[3] = sorted_points[2];  // Bottom-right
    }

    return corners;
}

cv::Mat build_minimap(std::vector<ParkingSpot>& parkingSpots){

    std::vector<cv::Point2f> all_vertices;
    for (ParkingSpot& spot : parkingSpots) {
        cv::RotatedRect rect = spot.rect;
        if(rect.size.area()>1) { // the if is needed because removing with the iterator produces rects with zero area
            cv::Point2f vertices[4];
            rect.points(vertices);
            for (int i = 0; i < 4; i++) {
                all_vertices.push_back(vertices[i]);
            }
        }
    }

    std::vector<cv::Point2f> hull;
    cv::convexHull(all_vertices, hull);

    // Draw the convex hull
    std::vector<std::pair<double, cv::Vec4f>> hullLines;    
    for (size_t i = 0; i < hull.size(); i++) {
        cv::Point2f p1 = hull[i];
        cv::Point2f p2 = hull[(i + 1) % hull.size()]; // Wrap around to form a closed hull
        double distance = cv::norm(p1-p2);
        hullLines.push_back(std::make_pair(distance, cv::Vec4f(p1.x, p1.y, p2.x, p2.y)));
    }

    // Sort the lines by their length in descending order
    std::sort(hullLines.begin(), hullLines.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });

    std::vector<double> ms;
    std::vector<double> bs;
    
    // Highlight the 4 longest lines in red
    // TODO: if hullLines.size() < 4 then throw error
    for (size_t i = 0; i < std::min(hullLines.size(), size_t(4)); i++) {
        auto& line = hullLines[i];
        double m = tan(get_segment_angular_coefficient(line.second)*CV_PI/180);
        double b = line.second[1] - m * line.second[0];
        ms.push_back(m);
        bs.push_back(b);
    }

    std::vector<cv::Point2f> hull_corners;
    // Check all pairs of lines for intersections
    for (size_t i = 0; i < ms.size(); ++i) {
        for (size_t j = i + 1; j < ms.size(); ++j) {
            double m1 = ms[i];
            double b1 = bs[i];
            double m2 = ms[j];
            double b2 = bs[j];

            // If lines have the same sign of slope, then we don't need to take their intersection
            if ((m1 < 0) == (m2 < 0)) {
                std::cout << "Lines " << i << " and " << j << " are parallel and do not intersect." << std::endl;
                continue;
            }

            // Calculate intersection point (x, y). 
            // Cannot use the function segments_intersect() because here we look at the extension of the hull lines
            double x = (b2 - b1) / (m1 - m2);
            double y = m1 * x + b1;

            hull_corners.push_back(cv::Point2f(x, y));
        }
    }

    // Sort the corner points
    std::vector<cv::Point2f> hull_corners_sorted = find_corners(hull_corners);

    int map_height = 250;
    int map_width = 450;
    cv::Size map_size(map_width,map_height);
    std::vector<cv::Point2f> to_hom_points = {cv::Point2f(0,map_height-1), cv::Point2f(0,-25), cv::Point2f(map_width-1,map_height-1), cv::Point2f(map_width-1,-25)};
    cv::Mat F = cv::getPerspectiveTransform(hull_corners_sorted, to_hom_points);

    cv::Mat minimap(map_size, CV_8UC3, cv::Scalar(255,255,255));
    
    double sum_angle;
    double avg_angle;
    std::vector<cv::RotatedRect> transformed_rects;
    std::vector<bool> occupancies;
    for(ParkingSpot spot: parkingSpots) {
        // Extract the vertices of the current RotatedRect
        cv::RotatedRect rect = spot.rect;
        occupancies.push_back(spot.occupied);
        cv::Point2f vertices[4];
        rect.points(vertices);

        // Prepare vectors to hold the original and transformed vertices
        std::vector<cv::Point2f> to_transform(vertices, vertices + 4);  // Collect vertices into a vector
        std::vector<cv::Point2f> transformed_vertices;

        // Apply perspective transformation
        cv::perspectiveTransform(to_transform, transformed_vertices, F);

        // Compute the minimum area rectangle from the transformed vertices
        cv::RotatedRect minrect = cv::minAreaRect(transformed_vertices);
        transformed_rects.push_back(minrect);
        vertices[4];
        minrect.points(vertices);
        for (int i = 0; i < 4; i++) {
            //cv::line(minimap, vertices[i], vertices[(i+1) % 4], cv::Scalar(0, 255, 0), 2);
        }
        sum_angle += minrect.angle;
    }

    avg_angle = sum_angle/parkingSpots.size();
    align_rects(transformed_rects,30);
    
    // Example larger image (e.g., 800x800 white background)
    cv::Mat large_image(300, 500, minimap.type(), cv::Scalar(255, 255, 255));  // A white image

    double offset_y = (large_image.rows-minimap.rows)/2;
    double offset_x  = (large_image.cols-minimap.cols)/2;
    for(int i = 0; i<transformed_rects.size(); i++) {
        cv::RotatedRect rect = transformed_rects[i];
        bool occupancy = occupancies[i];
        cv::RotatedRect to_print(cv::Point2f(rect.center.x+offset_x,rect.center.y+offset_y),cv::Size(60,20),(rect.size.aspectRatio()>1.4?avg_angle:-avg_angle));
        cv::Point2f vertices[4];
        to_print.points(vertices);
        for (int i = 0; i < 4; i++) {
            cv::line(large_image, vertices[i], vertices[(i+1) % 4], occupancy ? cv::Scalar(0, 0, 255) : cv::Scalar(255,0,0), 2);
        }
    }

    // Display the result
    cv::imshow("Centered Minimap", large_image);
    cv::waitKey(0);

    return large_image;
}

bool is_alone(cv::RotatedRect rect, std::vector<cv::RotatedRect> rects) {
    cv::RotatedRect extended = scale_rotated_rect(rect,1.5);
    for(const cv::RotatedRect other_rect:rects) {
        if(other_rect.center != rect.center && are_rects_overlapping(extended,other_rect)) {
            return false;
        }
    }

    return true;
}

// Function to shift a rotated rect along its longest direction by a given shift amount
cv::RotatedRect shift_along_longest_axis(const cv::RotatedRect& rect, float shift_amount, bool invert_direction) {
    // Find the longer dimension of the rectangle
    cv::Point2f vertices[4];
    rect.points(vertices);

    // Compute the longest axis direction
    cv::Point2f axis = (cv::norm(vertices[0] - vertices[1]) > cv::norm(vertices[1] - vertices[2])) ?
                        (vertices[1] - vertices[0]) : (vertices[2] - vertices[1]);

    // Normalize the axis vector to shift along its direction
    cv::Point2f normalized_axis = axis / cv::norm(axis);

    // Shift the center of the rotated rect along this axis
    cv::Point2f new_center = rect.center + shift_amount * (invert_direction ? -normalized_axis: normalized_axis);

    // Return a new rotated rect with the updated center
    return cv::RotatedRect(new_center, rect.size, rect.angle);
}

// Function to shift elements in vector2 if they overlap with elements in vector1
void resolve_overlaps(std::vector<cv::RotatedRect>& vector1, std::vector<cv::RotatedRect>& vector2, float shift_amount) {
    for (auto& rect1 : vector1) {
        for (auto& rect2 : vector2) {
            // Check if the two rectangles overlap
            while (computeIntersectionArea(rect1, rect2)>0.1) {
                // Shift rect2 along its longest axis until it no longer overlaps
                //rect1 = shift_along_longest_axis(rect1,shift_amount,true);
                rect2 = shift_along_longest_axis(rect2, shift_amount,false);
            }
        }
    }
}

// Function to scale a RotatedRect by a given scale factor
cv::RotatedRect scale_rotated_rect(const cv::RotatedRect& rect, float scale_factor) {
    // Scale the size (width and height) of the rotated rect
    cv::Size2f new_size(rect.size.width * scale_factor, rect.size.height * scale_factor);

    // Create a new rotated rect with the scaled size
    return cv::RotatedRect(rect.center, new_size, rect.angle);
}

std::pair<cv::RotatedRect, cv::RotatedRect> split_and_shift_rotated_rect(const cv::RotatedRect& rect) {
    cv::Point2f vertices[4];
    rect.points(vertices);
    float shift_amount = rect.size.width;
    // Find the midpoint along the longest side (between vertices[0] and vertices[1])
    cv::Point2f midpoint1 = (vertices[0] + vertices[1]) * 0.5;
    cv::Point2f midpoint2 = (vertices[2] + vertices[3]) * 0.5;

    // Calculate the shift direction based on the angle of the rectangle
    float angle_rad = (rect.angle+35) * CV_PI / 180.0;  // Convert angle to radians
    cv::Point2f shift_vector_x(shift_amount * std::cos(-angle_rad), shift_amount * std::sin(-angle_rad));  // Shift in x-direction
    cv::Point2f shift_vector_y(-shift_amount * std::sin(-angle_rad), shift_amount * std::cos(-angle_rad)); // Shift in y-direction

    // Shift the midpoints along the x and y axes
    cv::Point2f shifted_center1 = midpoint1 - shift_vector_x - shift_vector_y;
    cv::Point2f shifted_center2 = midpoint2 + shift_vector_x + shift_vector_y;

    // Create two new rotated rects, each with half the original width, and shifted
    cv::RotatedRect rect_part1(shifted_center1, cv::Size2f(rect.size.width, rect.size.height/2), rect.angle+35);
    cv::RotatedRect rect_part2(shifted_center2, cv::Size2f(rect.size.width , rect.size.height/2), rect.angle+35);
    
    return std::make_pair(rect_part1, rect_part2);
}

// Modified function to filter the first vector based on surrounding conditions
std::vector<cv::RotatedRect> filter_by_surrounding(const std::vector<cv::RotatedRect>& rects1, const std::vector<cv::RotatedRect>& rects2, cv::Mat image) {
    std::vector<cv::RotatedRect> filtered_rects;

    for (const auto& rect1 : rects1) {
        // Split rect1 into two equal parts along its longest direction
        auto [rect_part1, rect_part2] = split_and_shift_rotated_rect(rect1);
        
        // Scale both parts
        rect_part1 = scale_rotated_rect(rect_part1, 1.25);
        rect_part2 = scale_rotated_rect(rect_part2, 1.25);
        
        bool part1_overlap = false, part2_overlap = false;

        // Check if both parts overlap with any rect in rects2
        for (const auto& rect2 : rects2) {
            if (computeIntersectionArea(rect_part1, rect2) > 0) {
                part1_overlap = true;
            }
            if (computeIntersectionArea(rect_part2, rect2) > 0) {
                part2_overlap = true;
            }
            // If both parts overlap with at least one rect, we can discard rect1
            if (part1_overlap && part2_overlap) {
                break;
            }
        }

        // Only keep rect1 if not both parts overlap with any rect in rects2
        if (!(part1_overlap && part2_overlap) && rect1.size.area() > 1) {
            filtered_rects.push_back(rect1);
        }
    }

    return filtered_rects;
}

// Function to calculate the median of a vector
double compute_median(std::vector<double>& data) {
    if (data.empty()) return 0.0;
    std::sort(data.begin(), data.end());
    size_t n = data.size();
    if (n % 2 == 0) {
        return (data[n / 2 - 1] + data[n / 2]) / 2.0;
    } else {
        return data[n / 2];
    }
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


// Function to shrink a rotated rect
cv::RotatedRect shrink_rotated_rect(const cv::RotatedRect& rect, float shorten_percentage) {
    // Get the current size of the rectangle
    cv::Size2f size = rect.size;

    // Determine which side is the shortest and which is the longest
    float short_side = std::min(size.width, size.height);
    float long_side = std::max(size.width, size.height);

    // Shorten the longest side by the given percentage
    float new_long_side = long_side * (1.0f - shorten_percentage);

    // Set the new size, keeping the shortest side unchanged
    if (size.width > size.height) {
        size.width = new_long_side;
    } else {
        size.height = new_long_side;
    }

    // Return a new rotated rect with the updated size
    return cv::RotatedRect(rect.center, size, rect.angle);
}

// Function to get the right-most endpoint of a segment
cv::Point2f get_rightmost_endpoint(const cv::Vec4f& segment) {
    cv::Point2f p1(segment[0], segment[1]);  // First endpoint (x1, y1)
    cv::Point2f p2(segment[2], segment[3]);  // Second endpoint (x2, y2)

    // Compare the x-coordinates to find the right-most point
    return (p1.x > p2.x) ? p1 : p2;
}

// Function to extend a segment by a certain percentage of its length
cv::Vec4f extend_segment(const cv::Vec4f& seg, float extension_ratio) {
    cv::Point2f p1(seg[0], seg[1]), q1(seg[2], seg[3]);

    // Compute direction vector of the segment
    cv::Vec2f direction = cv::Vec2f(q1 - p1);
    float length = get_segment_length(seg);
    
    // Normalize the direction vector to unit length
    cv::Vec2f direction_normalized = direction / length;

    // Compute the extension length (25% of the segment length)
    float extension_length = length * extension_ratio;

    // Extend in both directions by converting to cv::Point2f for vector arithmetic
    cv::Point2f extended_p1 = p1 - cv::Point2f(direction_normalized[0], direction_normalized[1]) * extension_length;
    cv::Point2f extended_q1 = q1 + cv::Point2f(direction_normalized[0], direction_normalized[1]) * extension_length;

    // Return the new extended segment
    return cv::Vec4f(extended_p1.x, extended_p1.y, extended_q1.x, extended_q1.y);
}

void trim_if_intersect(cv::Vec4f& seg1, cv::Vec4f& seg2) {
    cv::Point2f intersection;
    if(segments_intersect(seg1, seg2, intersection)) {
        seg1[0] = intersection.x;
        seg1[1] = intersection.y;

        cv::Point2f right_endpoint(seg2[2], seg2[3]);
        cv::Point2f left_endpoint(seg2[0], seg2[1]);
        double norm_left = cv::norm(intersection-left_endpoint);
        double norm_right = cv::norm(intersection-right_endpoint);
        if(norm_left > norm_right) {
            seg2[2] = intersection.x;
            seg2[3] = intersection.y;
        }
        else {
            seg2[0] = intersection.x;
            seg2[1] = intersection.y;
        }
    }
}

// Function to check if two segments intersect (after extending)
bool segments_intersect(const cv::Vec4f& seg1, const cv::Vec4f& seg2, cv::Point2f& intersection) {

    // Extract points from the extended segments
    cv::Point2f p1(seg1[0], seg1[1]), q1(seg1[2], seg1[3]);
    cv::Point2f p2(seg2[0], seg2[1]), q2(seg2[2], seg2[3]);

    // Compute direction vectors for both segments
    cv::Vec2f r = cv::Vec2f(q1 - p1);  // Direction of extended_seg1
    cv::Vec2f s = cv::Vec2f(q2 - p2);  // Direction of extended_seg2

    float rxs = r[0] * s[1] - r[1] * s[0];
    cv::Vec2f qp = cv::Vec2f(p2 - p1);
    
    // Check if the lines are parallel
    if (std::fabs(rxs) < FLT_EPSILON) {
        return false;  // Parallel lines
    }

    float t = (qp[0] * s[1] - qp[1] * s[0]) / rxs;
    float u = (qp[0] * r[1] - qp[1] * r[0]) / rxs;

    // Check if the intersection happens within the segment bounds
    if (t >= 0 && t <= 1 && u >= 0 && u <= 1) {
        intersection = p1 + cv::Point2f(r[0] * t, r[1] * t);  // Find the intersection point
        return true;
    }
    return false;
}

// Function to move along the perpendicular direction from the right-most endpoint
cv::RotatedRect build_rotatedrect_from_movement(const cv::Vec4f& segment, const std::vector<cv::Vec4f>& segments, cv::Mat image) {
    // Rightmost endpoint of the original segment
    cv::Point2f right_endpoint(segment[2], segment[3]);
    cv::Point2f left_endpoint(segment[0], segment[1]);
    cv::Point2f midpoint = compute_midpoint(segment);

    // Compute the direction vector of the original segment
    cv::Vec2f direction = cv::Vec2f(right_endpoint - left_endpoint);
    float length = std::sqrt(direction[0] * direction[0] + direction[1] * direction[1]);
    length = get_segment_length(segment);
    // Normalize the direction vector
    direction /= length;

    double slope = tan(get_segment_angular_coefficient(segment)*CV_PI/180);
    double intercept =  segment[1] - slope * segment[0];

    cv::Point2f start;
    if(slope > 0) {
        start = left_endpoint+cv::Point2f(direction[0]*length*0.6,direction[1]*length*0.6);
    }
    else {
        // eventualmente lasciare solo left_endpoint, non cambia nulla se rotatedrect per template verticali non viene scalato dopo il match
        start = left_endpoint+cv::Point2f(direction[0]*length*0.25,direction[1]*length*0.25);
    }

    // Find the perpendicular direction (rotate by 90 degrees)
    cv::Vec2f perpendicular_direction(-direction[1], direction[0]);
    double search_length = std::min(2.5*length, 200.0);

    // Create the perpendicular segment from the rightmost endpoint
    cv::Point2f perp_end = start + cv::Point2f(perpendicular_direction[0] * search_length, perpendicular_direction[1] * search_length);

    // Initialize variables to store the closest intersection point
    cv::Point2f closest_intersection;
    float min_distance = std::numeric_limits<float>::max();
    bool found_intersection = false;
    cv::Vec4f closest_segment;
    // cv::circle(image,start,5,cv::Scalar(0,0,255));
    // cv::line(image, start, perp_end, cv::Scalar(0, 0, 255), 2, cv::LINE_AA); 
    // cv::line(image, start, perp_end, cv::Scalar(0, 0, 0), 2, cv::LINE_AA); 

    // Check intersection of the perpendicular segment with every other segment
    for (const auto& other_segment : segments) {
        cv::Point2f intersection;
        cv::Vec4f perp_vect = cv::Vec4f(start.x, start.y, perp_end.x, perp_end.y);
        cv::Vec4f extended_seg1 = extend_segment(other_segment, 0.4f);
        if (other_segment != segment && segments_intersect(extended_seg1, perp_vect, intersection)) {

            float dist = cv::norm(start-intersection);
            // last conditions to ensure that close segments of another parking slot line does not interfere
            // 
            if (dist > 22.5 && dist < min_distance) { // last conditions to ensure that close segments of another parking slot line does not interfere) { 
                min_distance = dist;
                closest_intersection = intersection;
                found_intersection = true;
                closest_segment = other_segment;
                // cv::imshow("projections", image);
                // cv::waitKey(0);
            }
        }
    }

    // If an intersection is found, use it to build the rotated rect
    if (found_intersection) {
        // Get the two endpoints of the second segment
        cv::Point2f endpoint1(closest_segment[0], closest_segment[1]);
        cv::Point2f endpoint2(closest_segment[2], closest_segment[3]);

        double length_other = get_segment_length(closest_segment);
        cv::Point2f destination_right(endpoint2.x, endpoint2.x*slope+intercept);

        cv::Point2f destination_bottom = destination_right + cv::Point2f(perpendicular_direction[0] * min_distance, perpendicular_direction[1] * min_distance);
        cv::RotatedRect bounding_box;
        if(slope>0) {
            bounding_box = cv::RotatedRect(left_endpoint,destination_right,destination_bottom);
        } 
        else {
            cv::Point2f destination_left((endpoint1.y-intercept)/slope,endpoint1.y);
            // or choose the other
            cv::Point2f destination_up = destination_left + cv::Point2f(perpendicular_direction[0]*min_distance, perpendicular_direction[1] * min_distance);
            //bounding_box = cv::RotatedRect(right_endpoint,left_endpoint,left_endpoint + cv::Point2f(perpendicular_direction[0] *min_distance, perpendicular_direction[1] *min_distance));
            bounding_box = cv::RotatedRect(right_endpoint,destination_left,destination_up);
        }
        //if(bounding_box.size.aspectRatio() > 1.5|| bounding_box.size.aspectRatio() < 1/1.5) {
        //    return shrink_rotated_rect(bounding_box, 0.8);
        //}
        cv::Point2f vertices[4];
        bounding_box.points(vertices);
        for (int i = 0; i < 4; i++) {
            // cv::line(image, vertices[i], vertices[(i+1) % 4], cv::Scalar(0, 0, 255), 2);
        }
        // cv::imshow("projections", image);
        // cv::waitKey(0);        
        return bounding_box;
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
            if (!merged[j] && are_rects_overlapping(rects[i], rects[j]) && are_rects_aligned(rects[i], rects[j],16)) {
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
std::vector<cv::Vec4f> filter_segments_near_top_right(const std::vector<cv::Vec4f>& segments, const cv::Size& image_size) {
    // Define the top-right corner of the image
    cv::Point2f top_right_corner(image_size.width - 1, 0);
    cv::Point2f start(850,0);
    cv::Point2f end(image_size.width-1,300);

    std::vector<cv::Point2f> hull;
    cv::convexHull(std::vector<cv::Point2f>{top_right_corner,start,end}, hull);

    std::vector<cv::Vec4f> filtered_segments;

    for (const auto& segment : segments) {
        cv::Point2f p1(segment[0],segment[1]);
        cv::Point2f p2(segment[2],segment[3]);
        cv::Point2f midpoint = compute_midpoint(segment);
        double result1 = cv::pointPolygonTest(hull, p1, false);  // False = no distance calculation needed
        double result2 = cv::pointPolygonTest(hull, p2, false);  // False = no distance calculation needed
        double result3 = cv::pointPolygonTest(hull,midpoint,false);

        if(result1<0 && result2<0 && result3 < 0) {
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

    cv::Mat element = cv::getStructuringElement( 
                        cv::MORPH_CROSS, cv::Size(3,3)); 

    // dil 2 erode 1
    cv::dilate(magnitude,adpt,element,cv::Point(-1,-1),4);
    cv::erode(adpt,adpt,element,cv::Point(-1,-1),3);

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
    return grad_magn_proc;
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

    height = 6;
    height = height + 4;
    width +=4;

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
        rotated_height = template_height;   // giusto così sennò si mangia un pezzo di riga e se matcha con la riga mangiata viene generato 
                                            // un rotatedrect lunghissimo
    }

    // Horizontal template and mask definition
    cv::Mat horizontal_template(template_height,template_width,CV_8U,cv::Scalar(0));
    cv::Mat horizontal_mask(template_height,template_width,CV_8U);

    // Build the template and mask
    for(int i = 0; i< horizontal_template.rows; i++) {
        for(int j = 0; j<horizontal_template.cols; j++) {
            if((!flipped ? i>2 && i < height-2 : j>2&&j<height-2) && (!flipped ? j>2 && j < width-2 : j>2&&j<width-2)) {
            horizontal_template.at<uchar>(i,j) = 255;
            horizontal_mask.at<uchar>(i,j) = 245;
            }
            else {
                horizontal_mask.at<uchar>(i,j) = 10;
            }
        }
    }

    rotation_center.y = template_height-1;
    rotation_center.x = 0;

    cv::Mat R = cv::getRotationMatrix2D(rotation_center, rotation_angle,1);
    cv::Mat rotated_template;
    cv::Mat rotated_mask;
    
    cv::warpAffine(horizontal_template,rotated_template,R,cv::Size(rotated_width,rotated_height));
    cv::warpAffine(horizontal_mask,rotated_mask,R,cv::Size(rotated_width,rotated_height));

    // cv::imshow("original template", horizontal_template);
    // cv::imshow("template", rotated_template);

    return std::vector<cv::Mat>{rotated_template,rotated_mask};
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

void nms(std::vector<cv::RotatedRect> &vec, std::vector<cv::RotatedRect> &elementsToRemove, double threshold, bool keep_smallest) {
    for (const auto& rect1 : vec) {
        for (const auto& rect2 : vec) {
            if (!(rect1.center.x == rect2.center.x && rect1.center.y == rect2.center.y) && (computeIntersectionArea(rect1, rect2) > threshold)) {
                if (keep_smallest ? rect1.size.area() < rect2.size.area() : rect1.size.area() > rect2.size.area()){
                    elementsToRemove.push_back(rect2);
                } else {
                    elementsToRemove.push_back(rect1);
                }
            }
        }
    }
}
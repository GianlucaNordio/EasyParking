#include "parkingSpotDetector.hpp"

/*
TODO: 
1. Preprocessing (pensare a come rendere invariante alle condizioni climatiche)
2. Generare meglio i template
3. Chiedere nel forum quanti parametri possiamo usare
4. Stesso size, angolo diverso: usare tm_result_unnormed come score, poi tra tutti quelli che overlappano per tipo l'80% tenere quello con score migliore
5. Dopo il punto 4, alla fine dei due cicli for, fare non maxima suppression. A quel punto usare NMS di opencv oppure prendere quello con area maggiore
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

    std::vector<double> pos_angles;
    std::vector<double> neg_angles;
    std::vector<double> pos_lengths;
    std::vector<double> neg_lengths;

    for (const cv::Vec4f& segment : segments) {
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

    double median_pos_angle = compute_median(pos_angles);
    double median_neg_angle = compute_median(neg_angles);
    double median_pos_width = compute_median(pos_lengths);
    double median_neg_width = compute_median(neg_lengths);

    std::cout << "Median positive angle: " << median_pos_angle << " degrees" << std::endl;
    std::cout << "Median negative angle: " << median_neg_angle << " degrees" << std::endl;
    std::cout << "Median width of positive angle lines: " << median_pos_width << std::endl;
    std::cout << "Median width of negative angle lines: " << median_neg_width << std::endl;

    // Display the result
    cv::imshow("Detected Line Segments", intermediate_results);
    cv::waitKey(0);

    cv::Mat element = cv::getStructuringElement( 
    cv::MORPH_CROSS, cv::Size(3,3)); 

    preprocessed = preprocess_find_parking_lines(image);
    cv::imshow("TM Input", preprocessed);
    cv::waitKey(0);

    // offsets from median values
    std::vector<int> angle_offsets = {-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7};
    std::vector<float> length_scales = {1.5,1.75};
    std::vector<cv::RotatedRect> list_boxes;
    std::vector<float> rect_scores(list_boxes.size(), -1); // Initialize scores with -1 for non-existing rects

    for(int l = 0; l<length_scales.size(); l++) {
        for(int k = 0; k<angle_offsets.size(); k++) {
            int template_width = median_pos_width*length_scales[l];
            int template_height = 4;
            double angle = -median_pos_angle+angle_offsets[k]; // negative

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

                cv::RotatedRect rotated_rect(center, cv::Size(template_width, template_height), -angle);

                // Check overlap with existing rects in list_boxes2
                bool overlaps = false;
                std::vector<size_t> overlapping_indices;

                for (size_t i = 0; i < list_boxes.size(); ++i) {
                    if (computeIntersectionArea(rotated_rect, list_boxes[i]) > 0.05) {
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
                    // Handle overlap case: check if the current rect's score is higher
                    bool add_current_rect = true;
                    for (size_t idx : overlapping_indices) {
                        if (score >= rect_scores[idx]) {
                            // The current rect has a lower or equal score, so don't add it
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
            int template_width = median_neg_width*length_scales[l];
            int template_height = 4;
            double angle = median_neg_angle+angle_offsets[k]; // negative

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

                cv::RotatedRect rotated_rect(center, cv::Size(template_width, template_height), angle);

                // Check overlap with existing rects in list_boxes2
                bool overlaps = false;
                std::vector<size_t> overlapping_indices;

                for (size_t i = 0; i < list_boxes2.size(); ++i) {
                    if (computeIntersectionArea(rotated_rect, list_boxes2[i]) > 0.05) {
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
                        if (score <= rect_scores2[idx]) {
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
    std::vector<cv::Vec4f> segments2;

    // Loop through all bounding boxes
    for (const auto& rect : merged_pos_rects) {
        // Convert the bounding box to a line
        cv::Vec4f line_segment = convert_rect_to_line(rect);
        segments2.push_back(line_segment);

        // Draw the line on the image
        cv::line(image, cv::Point2f(line_segment[0], line_segment[1]),
                 cv::Point2f(line_segment[2], line_segment[3]), cv::Scalar(255, 0, 0), 2);
    }

    // Loop through all bounding boxes
    for (const auto& rect : merged_neg_rects) {
        // Convert the bounding box to a line
        cv::Vec4f line_segment = convert_rect_to_line(rect);
        segments2.push_back(line_segment);

        // Draw the line on the image
        cv::line(image, cv::Point2f(line_segment[0], line_segment[1]),
                 cv::Point2f(line_segment[2], line_segment[3]), cv::Scalar(255, 0, 0), 2);
    }

	for(int i = 1; i<segments2.size(); i++)
    {
	    cv::Vec2f mq_param = get_segm_params(segments2[i]);
		cv::Point2f blue;
		cv::Point2f red;
		cv::Point2f green;
		cv::Point2f yellow;
		
		if(segments2[i][0]<segments2[i][2])//BLUE_START
		{
			blue = cv::Point2f(segments2[i][0],segments2[i][1]);
			red = cv::Point2f(segments2[i][2],segments2[i][3]);
            cv::circle(image,blue,10,cv::Scalar(255,0,0));
            cv::circle(image,red,10,cv::Scalar(0,0,255));
			
			cv::Vec2f doble = get_direction(segments2[i],true);//vector from blue to red
			
			green = cv::Point2f(segments2[i][2],segments2[i][3]);//red + doble_vector
			
			if(mq_param[0]>0)
			{
				//drawMarker(copied,Point2f(segments2[i][2]+doble[0]*0.7    - doble[1],segments2[i][3]+doble[1]+doble[0]*0.7   ),Scalar(0,255,255),MARKER_TILTED_CROSS,7,3);
				yellow = cv::Point2f(segments2[i][2] - doble[1]*0.35,segments2[i][3]+doble[0]*0.35   );
                cv::circle(image,yellow,10,cv::Scalar(0,255,0));
			}
			else
			{
				//drawMarker(copied,Point2f(segments2[i][2]+doble[0] + doble[1]*2   ,segments2[i][3]+doble[1]-doble[0]*2   ),Scalar(0,255,255),MARKER_TILTED_CROSS,7,3);
				yellow = cv::Point2f(segments2[i][2]+ doble[1]*1.7*0.5   ,segments2[i][3]-doble[0]*1.7*0.5   );
			}
		}
		else//RED_START
		{
			red = cv::Point2f(segments2[i][0],segments2[i][1]);
			blue = cv::Point2f(segments2[i][2],segments2[i][3]);

			cv::Vec2f doble = get_direction(segments2[i],false);

			green = cv::Point2f(segments2[i][0],segments2[i][1]+doble[1]);//red + doble_vector

			if(mq_param[0]>0)
			{
				yellow = cv::Point2f(segments2[i][0]- doble[1]*0.35   ,segments2[i][1]+doble[0]*0.35   );
			}
			else
			{
				yellow = cv::Point2f(segments2[i][0] + doble[1]*1.7*0.5   ,segments2[i][1]-doble[0]*1.7*0.5   );
			}
		}
		cv::circle(image,yellow,10,cv::Scalar(0,255,0));
        cv::imshow("Intersection of Rotated Rects", image);
        cv::waitKey(0);

		cv::RotatedRect sp = cv::RotatedRect(blue,red,yellow);
		//if()
		spots.push_back(sp);
	}


	std::vector<cv::RotatedRect> trimmed;
	for(int i = 0; i<spots.size()-1; i++)
	{
		cv::Point2f center1 = spots[i].center;
		bool found = false;
		for(int j = 0; j<trimmed.size() && !found; j++)
		{
			cv::Point2f center2 = trimmed[j].center;
			if((center1.x-center2.x)*(center1.x-center2.x) + (center1.y-center2.y)*(center1.y-center2.y) <800)//To reject overlapping rects
			{
				found = true;
			}
		}
		if(!found)
		{	
			if(spots[i].size.area()>2600 && spots[i].size.area()<13000)	//To reject small and large rects
				trimmed.push_back(spots[i]);
		}
	}

    // Apply NMS filtering
    std::vector<cv::RotatedRect> elementsToRemove;
    nms(trimmed, elementsToRemove);

    // Remove the elements determined by NMS filtering
    for (cv::RotatedRect element : elementsToRemove) {
        std::vector<cv::RotatedRect>::const_iterator iterator = elementIterator(trimmed, element);
        if (iterator != trimmed.cend()) {
            trimmed.erase(iterator);
        }
    }

    // Draw the rotated rectangles
    for (const auto& rect : trimmed) {
        // Get the 4 vertices of the rotated rectangle
        cv::Point2f vertices[4];
        rect.points(vertices);

        // Draw the rectangle
        for (int i = 0; i < 4; i++) {
            cv::line(image, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
        }
    }

    cv::imshow("Intersection of Rotated Rects", image);
    cv::waitKey(0);

	return parkingSpots;
}

double compute_median(std::vector<double>& data) {
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

cv::Vec2f get_segm_params(cv::Vec4f segm)
{
	float m = (segm[1] - segm[3])/(segm[0] - segm[2]);
    float q = segm[1] - m*segm[0];

	return cv::Vec2f(m,q);
}

cv::Vec2f get_direction(cv::Vec4f segm,bool blueStart){
	int offset = 0;
	if(!blueStart)
		offset = 2;
	return cv::Vec2f(segm[(2+offset)%4] - segm[(0+offset)%4], segm[(3+offset)%4] - segm[(1+offset)%4]);
}


// Helper function to check if two rotated rectangles overlap
bool are_rects_overlapping(const cv::RotatedRect& rect1, const cv::RotatedRect& rect2) {
    std::vector<cv::Point2f> intersection_points;
    int intersection_status = cv::rotatedRectangleIntersection(rect1, rect2, intersection_points);
    return intersection_status == cv::INTERSECT_FULL || intersection_status == cv::INTERSECT_PARTIAL;
}

// Helper function to check if two rotated rectangles are aligned (same angle within a tolerance)
bool are_rects_aligned(const cv::RotatedRect& rect1, const cv::RotatedRect& rect2, float angle_tolerance = 5) {
    return std::abs(rect1.angle - rect2.angle) <= angle_tolerance;
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
            if (!merged[j] && are_rects_overlapping(rects[i], rects[j]) && are_rects_aligned(rects[i], rects[j])) {
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
    return grad_magn+grad_magn_proc;
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
            horizontal_template.at<uchar>(i,j) = 200;
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
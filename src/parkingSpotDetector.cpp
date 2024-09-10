#include "parkingSpotDetector.hpp"

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

    cv::Mat preprocessed = preprocess(image);
    cv::Mat intermediate_results = image.clone();
    
    std::vector<int> angles = {-5,-6,-7,-8,-9, -10, -11, -12, -13,-14,-15,-16};
    std::vector<float> scales = {1,0.75,0.5};
    std::vector<cv::RotatedRect> list_boxes;

    for(int l = 0; l<scales.size(); l++) {
        for(int k = 0; k<angles.size(); k++) {
            // Template size
            int template_height = 5*scales[l];
            int template_width = 100*scales[l];

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

            // Rotate and flip the template
            cv::Mat flipped;
            cv::Mat flipped_mask;
            cv::flip(horizontal_template,flipped,-1);
            cv::flip(horizontal_mask,flipped_mask,-1);
            cv::Mat R = cv::getRotationMatrix2D(cv::Point2f(0,template_height-1),angles[k],1);
            cv::Mat rotated_template;
            cv::Mat rotated_mask;

            float rotated_width = template_width*cos(-angles[k]*CV_PI/180)+template_height;
            float rotated_height = template_width*sin(-angles[k]*CV_PI/180)+template_height;

            cv::warpAffine(horizontal_template,rotated_template,R,cv::Size(rotated_width,rotated_height));
            cv::warpAffine(horizontal_mask,rotated_mask,R,cv::Size(rotated_width,rotated_height));

            if(k == 0) {
                    cv::imshow("Horizontal template", horizontal_template);
                    cv::imshow("Rotated template", rotated_template);
            }

            cv::Mat tm_result_unnorm;
            cv::Mat tm_result;
            cv::matchTemplate(preprocessed,rotated_template,tm_result_unnorm,cv::TM_SQDIFF,rotated_mask);
            cv::normalize( tm_result_unnorm, tm_result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );
        
            // cv::imshow("TM Result", tm_result);
            // cv::waitKey(0);

            // Finding local minima
            cv::Mat eroded;
            std::vector<cv::Point> minima;
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(rotated_width, rotated_height));
            cv::erode(tm_result, eroded, kernel);
            cv::Mat localMinimaMask = (tm_result == eroded) & (tm_result <= 0.065);

            // Find all non-zero points (local minima) in the mask
            cv::findNonZero(localMinimaMask, minima);

            // Draw bboxes of the found lines
            for (const cv::Point& pt : minima) {
                // Get center of the bbox to draw the rotated rect
                cv::Point center;
                center.x = pt.x+rotated_width/2;
                center.y = pt.y+rotated_height/2;

                cv::RotatedRect rotatedRect(center, cv::Size(template_width,template_height), -angles[k]);
                list_boxes.push_back(rotatedRect);

                //Draw the rotated rectangle using lines between its vertices
                cv::Point2f vertices[4];
                rotatedRect.points(vertices);
                for (int i = 0; i < 4; i++) {
                    cv::line(intermediate_results, vertices[i], vertices[(i+1) % 4], cv::Scalar(0, 255, 0), 2);
                }
            }
        }
        cv::imshow("with lines", intermediate_results);
        cv::waitKey(0);
    }

    // Manually filter the rectangles based on the distance to the top-right corner
    cv::Point2f topRightCorner(image.cols - 1, 0);
    double distanceThreshold = 300.0;
    std::vector<cv::Point2f> filtered_verts;
    for (const auto& rect : list_boxes) {
        double dist =  std::sqrt(std::pow(rect.center.x - topRightCorner.x, 2) + std::pow(rect.center.y - topRightCorner.y, 2));;
        if (dist > distanceThreshold) {
            cv::Point2f vertices[4];
            rect.points(vertices);
            for (int i = 0; i < 4; i++) {
                cv::line(intermediate_results, vertices[i], vertices[(i+1) % 4], cv::Scalar(0, 255, 0), 2);
                filtered_verts.push_back(vertices[i]);
            }
        }
    }

    // Convex hull of bouding boxes that have been found
    std::vector<cv::Point2f> hull;
    cv::convexHull(filtered_verts, hull);

    // Draw the convex hull and save its lines with their lengths so that we can sort them based on length
    std::vector<std::pair<double, std::pair<cv::Point2f, cv::Point2f>>> hullLines;   
    for (size_t i = 0; i < hull.size(); i++) {
        // Line endpoints
        cv::Point2f p1 = hull[i];
        cv::Point2f p2 = hull[(i + 1) % hull.size()];

        //Draw the line
        cv::line(intermediate_results, p1, p2, cv::Scalar(0, 255, 0), 2);
        
        // Put its length in the vector of pairs
        double length = cv::norm(p1-p2);
        hullLines.push_back(std::make_pair(length, std::make_pair(p1, p2)));
    }

    // Sort the lines by their length, longest ones are first
    std::sort(hullLines.begin(), hullLines.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });
    
    // Compute mathematical properties of the longest lines
    std::vector<double> ms; // angular coefficients
    std::vector<double> bs; // intercepts
    for (size_t i = 0; i < std::min(hullLines.size(), size_t(5)); i++) {
        auto& line = hullLines[i];
        double m = static_cast<double>(line.second.second.y - line.second.first.y) / (line.second.second.x - line.second.first.x);
        double b = line.second.first.y - m * line.second.first.x;
        ms.push_back(m);
        bs.push_back(b);
        
        cv::line(intermediate_results, line.second.first, line.second.second, cv::Scalar(0, 0, 255), (i+1)*(i+1));
    }

    // Extend the longest lines and find their intersection. Such intersection will be the points to be used to find the homography
    std::vector<cv::Point2f> hom_points;
    for (size_t i = 0; i < ms.size(); ++i) {
        for (size_t j = i + 1; j < ms.size(); ++j) {
            double m1 = ms[i];
            double b1 = bs[i];
            double m2 = ms[j];
            double b2 = bs[j];

            // Since we consider only the longest lines, they are considered to be parallel if they have the same angular coefficient
            if ((m1 < 0) == (m2 < 0)) {
                continue;
            }

            // Intersection point
            double x = (b2 - b1) / (m1 - m2);
            double y = m1 * x + b1;

            // If the intersection point is not inside the image, then put it inside
            if(x<0) x = 0;
            if(x >= image.cols) x = image.cols-1;
            if(y <0) y = 0;
            if(y >= image.rows) y = image.rows -1;
            
            // Save the intersection point
            hom_points.push_back(cv::Point2f(x, y));
        }
    }

    // Distance threshold to keep only one intersection point if two are too close
    float pointsDistanceThreshold = 100.f;

    // Remove intersection points that are too close
    std::vector<cv::Point2f> filteredPoints = removeClosePoints(hom_points, pointsDistanceThreshold);

    // Iterate over the points to show the corners of the "extended convex jull"
    for (const auto& point : filteredPoints) {
            cv::circle(intermediate_results, cv::Point(static_cast<int>(point.x), static_cast<int>(point.y)), 5, cv::Scalar(0, 0, 255), -1);
            std::cout << point << std::endl; // Debug
    }

    cv::imshow("Image with homography points", intermediate_results);
    cv::waitKey(0);

    // Compute the homography
    std::vector<cv::Point2f> to_hom_points = {cv::Point2f(1199,60), cv::Point2f(1099,799), cv::Point2f(20,60), cv::Point2f(100,799)};
    cv::Mat F = cv::findHomography(filteredPoints, to_hom_points);
    
    // Apply the homography to the original image and the preprocessed one
    cv::Mat result_original;
    cv::Mat result_preproccesed;
    cv::warpPerspective(image, result_original, F, cv::Size(1200,800));
    cv::warpPerspective(preprocessed, result_preproccesed, F, cv::Size(1200,800));
    cv::imshow("result", result_preproccesed+preprocess(result_original));
    cv::waitKey(0);

    // Preprocess the transformed image
    cv::Mat result_gs;
    cv::cvtColor(result_original,result_gs,cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(result_gs,result_gs,cv::Size(3,3),30);

    cv::Mat adaptivethold;
    cv::adaptiveThreshold(result_gs, adaptivethold, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 15, 5);    
    cv::imshow("adaptive tholded", adaptivethold);
    cv::waitKey(0);

    cv::Mat gammatf = applyGammaTransform(result_gs, 0.4);
    cv::imshow("gamma homo", gammatf);

    cv::Mat gx;
    cv::Sobel(result_gs, gx, CV_16S, 1,0);

    cv::Mat gy;
    cv::Sobel(result_gs, gy, CV_16S, 0,1);

    cv::Mat abs_grad_x;
    cv::Mat abs_grad_y;
    cv::convertScaleAbs(gx, abs_grad_x);
    cv::convertScaleAbs(gy, abs_grad_y);

    cv::Mat grad_magn;
    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad_magn);
    cv::imshow("grad magn homo", grad_magn);

    cv::Mat element = cv::getStructuringElement( 
                        cv::MORPH_RECT, cv::Size(3,3)); 

    cv::Mat grad_magn_thold;
    cv::threshold(grad_magn,grad_magn_thold,20,255,cv::THRESH_BINARY);
    cv::erode(grad_magn_thold,grad_magn_thold,element,cv::Point(-1,-1),1);
    cv::dilate(grad_magn_thold,grad_magn_thold,element,cv::Point(-1,-1),2);
    cv::erode(grad_magn_thold,grad_magn_thold,element,cv::Point(-1,-1),1);
    cv::imshow("grad magn homo thold", grad_magn_thold);
    cv::waitKey(0);

std::vector<int> angles_2 = {-35,-36,-37,-38,-39,-40, -43,-45,-47,-52};
    std::vector<float> scales_2 = {0.85,0.9,1};
    std::vector<std::pair<cv::RotatedRect, double>> list_boxes_2;
    for(int l = 0; l<scales_2.size(); l++) {
        for(int k = 0; k<angles_2.size(); k++) {
            for(int f = 0; f<2; f++) {
                // Template size
                int line_width = 10;
                int template_height = 90*scales_2[l];
                int template_width = 145*scales_2[l];

                // Horizontal template and mask definition
                cv::Mat horizontal_template(template_height,template_width,CV_8U,cv::Scalar(0));
                cv::Mat horizontal_mask(template_height,template_width,CV_8U,cv::Scalar(0));

                for(int i = 0; i< horizontal_template.rows; i++) {
                    for(int j = 0; j<horizontal_template.cols; j++) {
                        if(((i<line_width) 
                            || (j > template_width-line_width || j < line_width) 
                            || (i > (template_height-line_width)))){
                            horizontal_template.at<uchar>(i,j) = 220;
                            horizontal_mask.at<uchar>(i,j) = 254;
                        }
                        else {
                            horizontal_mask.at<uchar>(i,j) = 1;
                        }
                    }
                }


                // Rotate
                cv::Mat R = cv::getRotationMatrix2D(cv::Point2f(0,template_height),angles_2[k],1);
                cv::Mat rotated_template;
                cv::Mat rotated_mask;

                float rotated_width = template_width*cos(-angles_2[k]*CV_PI/180)+template_height;
                float rotated_height = template_width*sin(-angles_2[k]*CV_PI/180)+template_height;

                cv::warpAffine(horizontal_template,rotated_template,R,cv::Size(rotated_width,rotated_height));
                cv::warpAffine(horizontal_mask,rotated_mask,R,cv::Size(rotated_width,rotated_height));

                // Flip
                cv::Mat flipped;
                cv::Mat flipped_mask;
                cv::flip(rotated_template,flipped,1);
                cv::flip(rotated_mask,flipped_mask,1);

                if(k == 0) {
                        cv::imshow("Horizontal template", flipped);
                        cv::imshow("Rotated template", flipped_mask);
                }

                if(f == 1) {
                    rotated_template = flipped;
                    rotated_mask = flipped_mask;
                }

                cv::Mat tm_result_unnorm;
                cv::Mat tm_result;
                cv::matchTemplate(adaptivethold,rotated_template,tm_result_unnorm,cv::TM_SQDIFF,rotated_mask);
                cv::normalize( tm_result_unnorm, tm_result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );
            
                // cv::imshow("homo TM Result", tm_result);
                // cv::waitKey(0);

                // Finding local minima
                cv::Mat eroded;
                std::vector<cv::Point> minima;
                cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(rotated_height*0.5, rotated_height*0.5));
                cv::erode(tm_result, eroded, kernel);
                cv::Mat localMinimaMask = (tm_result == eroded) & (tm_result < 0.2);
                // cv::imshow("homo TM Result eroded", eroded);
                // cv::waitKey(0);

                // Find all non-zero points (local minima) in the mask
                cv::findNonZero(localMinimaMask, minima);

                // Draw bboxes of the found lines
                for (const cv::Point& pt : minima) {
                    // Get center of the bbox to draw the rotated rect
                    cv::Point center;
                    center.x = pt.x+rotated_width/2;
                    center.y = pt.y+rotated_height/2;

                    cv::RotatedRect rotatedRect(center, cv::Size(template_width,template_height), -angles_2[k] - (f == 1 ? 90 : 0));
                    list_boxes_2.push_back(std::pair(rotatedRect, tm_result_unnorm.at<double>(pt)));

                    //Draw the rotated rectangle using lines between its vertices
                    cv::Point2f vertices[4];
                    rotatedRect.points(vertices);
                    /* for (int i = 0; i < 4; i++) {
                        cv::line(result_original, vertices[i], vertices[(i+1) % 4], cv::Scalar(0, 255, 0), 2);
                    } */
                }
            }
        }
        /* cv::imshow("homo with lines", result_original);
        cv::waitKey(0); */
    }


    // Filter out the boxes that have more than half of their content black
    filterBoundingBoxes(grad_magn_thold, list_boxes_2);

    // Apply NMS filtering to the boxes found
    std::vector<std::pair<cv::RotatedRect, double>> elementsToRemove;

    for (const auto& box : list_boxes_2) {
        for (const auto& box2 : list_boxes_2) {
            cv::RotatedRect rect1 = box.first;
            cv::RotatedRect rect2 = box2.first;

            double score1 = box.second;
            double score2 = box2.second;

            if (rect1.center.x == rect2.center.x && rect1.center.y == rect2.center.y && score1 == score2) {
                std::cout << "same rect" << std::endl;
            } else if (computeIntersectionArea(rect1, rect2) > 0.4) {
                if (score1 >= score2) {
                    elementsToRemove.push_back(box2);
                } else {
                    elementsToRemove.push_back(box);
                }
            }
        }
    }

    // Rimuovi tutti gli elementi raccolti
    for (std::pair element : elementsToRemove) {
        std::vector<std::pair<cv::RotatedRect, double>>::const_iterator iterator = elementIterator(list_boxes_2, element);
        if (iterator != list_boxes_2.cend()) {
            list_boxes_2.erase(iterator);
        }
    }

    // Create final_boxes that will contain the final boxes
    std::vector<cv::RotatedRect> final_boxes;
    for (const auto& box : list_boxes_2) {
        final_boxes.push_back(box.first);
    }


    // Draw the remaining bounding boxes on the image
    for (const auto& box : final_boxes) {
        cv::Point2f vertices[4];
        box.points(vertices);
        for (int i = 0; i < 4; i++) {
            cv::line(result_original, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 0, 255), 2);
        }
    }

    // Display the result
    cv::imshow("Filtered Bounding Boxes", result_original);
    cv::waitKey(0);

    cv::Mat inverseMatrix;
    cv::invert(F, inverseMatrix);

    // Iterate over each RotatedRect, inverse transform it, and draw it on the original image
    for (const auto& rRect : final_boxes) {
        // Get the four corner points of the RotatedRect in the transformed image
        cv::Point2f rectPoints[4];
        rRect.points(rectPoints);

        // Transform the points back to the original image
        std::vector<cv::Point2f> originalPoints;
        cv::perspectiveTransform(std::vector<cv::Point2f>(rectPoints, rectPoints + 4), originalPoints, inverseMatrix);

        // Convert the vector of points to an array of Points
        std::vector<cv::Point> originalPointsInt;
        for (const auto& pt : originalPoints) {
            originalPointsInt.push_back(cv::Point(cvRound(pt.x), cvRound(pt.y)));
        }

        // Draw the RotatedRect on the original image
        cv::polylines(image, originalPointsInt, true, cv::Scalar(0, 255, 0), 2);
    }

    // Show the result
    cv::imshow("Inverse Transformed RotatedRects", image);
    cv::waitKey(0);


    return parkingSpots;
}

double computeIntersectionArea(const cv::RotatedRect& rect1, const cv::RotatedRect& rect2) {
    // Converti i RotatedRect in vettori di punti (poligoni)
    std::vector<cv::Point2f> points1, points2;
    cv::Point2f vertices1[4], vertices2[4];

    double area = rect1.size.area();
    
    rect1.points(vertices1);
    rect2.points(vertices2);
    
    for (int i = 0; i < 4; i++) {
        points1.push_back(vertices1[i]);
        points2.push_back(vertices2[i]);
    }

    // Calcola l'intersezione tra i due poligoni
    std::vector<cv::Point2f> intersection;
    double intersectionArea = cv::intersectConvexConvex(points1, points2, intersection) / area;

    std::cout << "Intersection area: " << intersectionArea << std::endl;

    return intersectionArea;
}

std::vector<std::pair<cv::RotatedRect, double>>::const_iterator elementIterator(
    const std::vector<std::pair<cv::RotatedRect, double>>& vec,
    const std::pair<cv::RotatedRect, double>& elem) 
{
    for (auto it = vec.cbegin(); it != vec.cend(); ++it) {
        if (it->second == elem.second &&
            it->first.center.x == elem.first.center.x &&
            it->first.center.y == elem.first.center.y) 
        {
            return it; // Restituiamo l'iteratore all'elemento
        }
    }
    return vec.cend(); // Restituiamo end() se l'elemento non è stato trovato
}

bool isDarkerThanThreshold(const cv::Mat& image, const cv::RotatedRect& box, double threshold) {
    // Create a mask for the RotatedRect
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);

    // Get the vertices of the RotatedRect
    cv::Point2f vertices[4];
    box.points(vertices);

    // Fill the mask with white color where the bounding box is
    std::vector<cv::Point> pts;
    for (int i = 0; i < 4; i++) {
        pts.push_back(vertices[i]);
    }
    cv::fillConvexPoly(mask, pts, cv::Scalar(255));

    // Count the non-black pixels inside the bounding box
    cv::Mat roi;
    image.copyTo(roi, mask);

    int totalPixels = cv::countNonZero(mask);
    int blackPixels = totalPixels - cv::countNonZero(roi);

    // Return true if more than half of the pixels inside the bounding box are black
    return blackPixels > (totalPixels * threshold);
}

void filterBoundingBoxes(cv::Mat& image, std::vector<std::pair<cv::RotatedRect, double>>& boxes) {
    boxes.erase(std::remove_if(boxes.begin(), boxes.end(),
                               [&image](const std::pair<cv::RotatedRect, double>& boxPair) {
                                   return isDarkerThanThreshold(image, boxPair.first, 0.85);
                               }),
                boxes.end());
}

// Function to calculate the Euclidean distance between two points
float calculateDistance(const cv::Point2f& p1, const cv::Point2f& p2) {
    return std::sqrt(std::pow(p2.x - p1.x, 2) + std::pow(p2.y - p1.y, 2));
}

// Function to remove points that are very close together
std::vector<cv::Point2f> removeClosePoints(const std::vector<cv::Point2f>& points, float distanceThreshold) {
    std::vector<cv::Point2f> result;

    for (size_t i = 0; i < points.size(); ++i) {
        bool isTooClose = false;

        // Compare the current point with the points already in the result vector
        for (size_t j = 0; j < result.size(); ++j) {
            if (calculateDistance(points[i], result[j]) < distanceThreshold) {
                isTooClose = true;
                break;
            }
        }

        // If the point is not too close to any point in the result, add it
        if (!isTooClose) {
            result.push_back(points[i]);
        }
    }

    return result;
}

/*  Function that preprocesses the original images of sequence0.
    Pipeline:
        - Bilateral filter to smooth out some road imperfections and keep white lines and the edges of the space between lines of parking slots
        - Gradient magnitude to highlight the outline of the parking slots as much as possible across the different images of sequence0
        - Double dilation to fill in weak and double lines
        - Erosion to reduce the thickness of the lines
*/

cv::Mat preprocess(const cv::Mat& src) {
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

    cv::Mat grad_magn;
    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad_magn);

    cv::Mat edges;
    cv::Canny(grad_magn, edges,150, 400);

    cv::Mat element = cv::getStructuringElement( 
                        cv::MORPH_CROSS, cv::Size(3,3)); 

    cv::dilate(edges,edges,element,cv::Point(-1,-1),2);
    cv::erode(edges,edges,element,cv::Point(-1,-1),1);

    return edges;
}

/*  Function that computes the Gamma transform of the desired image. */
cv::Mat applyGammaTransform(const cv::Mat& src, double gamma) {
    // Lookup table for the gamma transform
    cv::Mat lookupTable(1, 256, CV_8U);
    uchar* p = lookupTable.ptr();

    for (int i = 0; i < 256; ++i) {
        p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
    }

    // Apply table to the desired image
    cv::Mat dst;
    cv::LUT(src, lookupTable, dst);

    return dst;
}
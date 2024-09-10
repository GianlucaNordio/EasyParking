#include "parkingSpotDetector.hpp"

/*
TODO: 
1. Preprocessing (pensare a come rendere invariante alle condizioni climatiche)
2. Generare meglio i template
3. Chiedere nel forum quanti parametri possiamo usare
4. Stesso size, angolo diverso: usare tm_result_unnormed come score, poi tra tutti quelli che overlappano per tipo l'80% tenere quello con score migliore
5. Dopo il punto 4, alla fine dei due cicli for, fare non maxima suppression. A quel punto usare NMS di opencv oppure prendere quello con area maggiore
*/

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

            cv::warpAffine(flipped,rotated_template,R,cv::Size(rotated_width,rotated_height));
            cv::warpAffine(flipped_mask,rotated_mask,R,cv::Size(rotated_width,rotated_height));

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

/* Procedimento: 
    1-Filtro bbox in alto a destra (unica cosa che si può fare a mano)
    2-Convex hull con i bbox rimanenti
    3-Trovo le linee più lunghe del convex hull, le prolungo e calcolo le loro intersezioni ottenendo 4 punti da proiettare tramite omografia
    4-Faccio l'omografia
    Problemino: 
        Alcune linee dei convex hull sono consecutive e quasi parallele, quindi 2 delle 4 linee più lunghe possono appartenere allo stesso "lato".
        Quello che faccio ora è selezionare le 5 linee più lunghe, estenderle, trovare intersezioni e togliere quei punti di intersezione 
        che sono troppo vicini tra loro. L'immagine 3 non ha questo problema, pertanto la soluzione attuale scompiglia tutto.
        La vera soluzione sarebbe questa: se due delle 4 linee più lunghe sono consecutive e quasi parallele, allora le considero come una.
*/

    // Manually filter the rectangles based on the distance to the top-right corner (allowed)
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

    // Get convex hull of every line that has been found
    std::vector<cv::Point2f> hull;
    cv::convexHull(filtered_verts, hull);

    // Draw the convex hull and save its lines
    std::vector<std::pair<double, std::pair<cv::Point2f, cv::Point2f>>> hullLines;    
    for (size_t i = 0; i < hull.size(); i++) {
        cv::line(intermediate_results, hull[i], hull[(i + 1) % hull.size()], cv::Scalar(0, 255, 0), 2);
        cv::Point2f p1 = hull[i];
        cv::Point2f p2 = hull[(i + 1) % hull.size()]; // Wrap around to form a closed hull
        double distance = cv::norm(p1-p2);
        hullLines.push_back(std::make_pair(distance, std::make_pair(p1, p2)));
    }

    // Sort the lines by their length in descending order
    std::sort(hullLines.begin(), hullLines.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });
    
    // Highlight the 4 longest lines in red
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

    // Find the points to transform by extending the longest lines and finding their intersection
    std::vector<cv::Point2f> hom_points;
    for (size_t i = 0; i < ms.size(); ++i) {
        for (size_t j = i + 1; j < ms.size(); ++j) {
            double m1 = ms[i];
            double b1 = bs[i];
            double m2 = ms[j];
            double b2 = bs[j];

            std::cout << "Lines " << i << " m: "<< m1 << " and " << j << " m: " << m2 << std::endl;
            // Check if lines are parallel (have the same slope)
            if ((m1 < 0) == (m2 < 0)) {
                continue;
            }

            // Calculate intersection point (x, y)
            double x = (b2 - b1) / (m1 - m2);
            double y = m1 * x + b1;

            // Check if the intersection point is inside the image
            if(x<0) x = 0;
            if(x >= image.cols) x = image.cols-1;
            if(y <0) y = 0;
            if(y >= image.rows) y = image.rows -1;
            
            hom_points.push_back(cv::Point2f(static_cast<int>(x), static_cast<int>(y)));
        }
    }

    // Distance threshold (adjust as needed)
    float pointsDistanceThreshold = 100.f;

    // Remove points that are too close together
    std::vector<cv::Point2f> filteredPoints = removeClosePoints(hom_points, pointsDistanceThreshold);

    // Iterate over the points to show the corners
    for (const auto& point : filteredPoints) {
            cv::circle(intermediate_results, cv::Point(static_cast<int>(point.x), static_cast<int>(point.y)), 5, cv::Scalar(0, 0, 255), -1);
            std::cout << point << std::endl;
    }
    cv::imshow("image with homography points", intermediate_results);
    cv::waitKey(0);

    std::vector<cv::Point2f> to_hom_points = {cv::Point2f(1199,60), cv::Point2f(1099,799), cv::Point2f(20,60), cv::Point2f(100,799)};
    cv::Mat F = cv::findHomography(filteredPoints, to_hom_points);
    cv::Mat result_original;
    cv::Mat result_preproccesed;
    cv::warpPerspective(image, result_original, F, cv::Size(1200,800));
    cv::warpPerspective(preprocessed, result_preproccesed, F, cv::Size(1200,800));
    cv::imshow("result", result_preproccesed+preprocess(result_original));
    cv::waitKey(0);

    cv::Mat result_gs;
    cv::cvtColor(result_original,result_gs,cv::COLOR_BGR2GRAY);

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
            // Template size
            int line_width = 8;
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

            cv::Mat tm_result_unnorm;
            cv::Mat tm_result;
            cv::matchTemplate(adaptivethold,rotated_template ,tm_result_unnorm,cv::TM_SQDIFF,rotated_mask);
            cv::normalize( tm_result_unnorm, tm_result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );
        
            cv::imshow("homo TM Result", tm_result);
            cv::waitKey(0);

            // Finding local minima
            cv::Mat eroded;
            std::vector<cv::Point> minima;
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(rotated_height*0.5, rotated_height*0.5));
            cv::erode(tm_result, eroded, kernel);
            cv::Mat localMinimaMask = (tm_result == eroded) & (tm_result < 0.2);
            cv::imshow("homo TM Result eroded", eroded);
            cv::waitKey(0);

            // Find all non-zero points (local minima) in the mask
            cv::findNonZero(localMinimaMask, minima);

            // Draw bboxes of the found lines
            for (const cv::Point& pt : minima) {
                // Get center of the bbox to draw the rotated rect
                cv::Point center;
                center.x = pt.x+rotated_width/2;
                center.y = pt.y+rotated_height/2;

                cv::RotatedRect rotatedRect(center, cv::Size(template_width,template_height), -angles_2[k]);
                list_boxes_2.push_back(std::pair(rotatedRect, tm_result_unnorm.at<double>(pt)));

                //Draw the rotated rectangle using lines between its vertices
                cv::Point2f vertices[4];
                rotatedRect.points(vertices);
                for (int i = 0; i < 4; i++) {
                    cv::line(result_original, vertices[i], vertices[(i+1) % 4], cv::Scalar(0, 255, 0), 2);
                }
            }
        }
        cv::imshow("homo with lines", result_original);
        cv::waitKey(0);
    }

    std::cout << "Arrivato alla nms" << std::endl;

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

    // Filter out the boxes that have more than half of their content black
    filterBoundingBoxes(grad_magn_thold, final_boxes);


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

    return parkingSpots;
}

bool isMoreThanHalfBlack(const cv::Mat& image, const cv::RotatedRect& box) {
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
    return blackPixels > (totalPixels * 0.85);
}

void filterBoundingBoxes(cv::Mat& image, std::vector<cv::RotatedRect>& boxes) {
    boxes.erase(std::remove_if(boxes.begin(), boxes.end(),
                               [&image](const cv::RotatedRect& box) {
                                   return isMoreThanHalfBlack(image, box);
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

cv::Mat preprocess(const cv::Mat& src) {
    cv::Mat gaussianFilteredImage;
    cv::GaussianBlur(src,gaussianFilteredImage,cv::Size(3,3),20);
    cv::Mat filteredImage;
    cv::bilateralFilter(src, filteredImage, -1, 40, 10);

    cv::Mat gs;
    cv::cvtColor(filteredImage, gs, cv::COLOR_BGR2GRAY);

    cv::imshow("grayscale", gs);
    cv::waitKey(0);

    cv::Mat stretched = contrastStretchTransform(gs);
    //cv::imshow("stretched", stretched);
    //cv::waitKey(0);

    // Set the gamma value
    double gammaValue = 1.25; // Example gamma value

    // Apply gamma transformation
    cv::Mat gammaCorrected1 = applyGammaTransform(gs, gammaValue);
    //cv::imshow("gamma tf1", gammaCorrected1);
    //cv::waitKey(0);

    gammaValue = 2;
    cv::Mat gammaCorrected2 = applyGammaTransform(gammaCorrected1, gammaValue);
    //cv::imshow("gamma tf2", gammaCorrected2);
    //cv::waitKey(0);

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

    cv::Mat equalized;
    cv::equalizeHist(gammaCorrected2,equalized);
    //cv::imshow("equalized", equalized);
    //cv::waitKey(0);

    cv::Mat laplacian;
    cv::Laplacian(equalized, laplacian, CV_32F);  // Use CV_32F to avoid overflow

    // Compute the absolute value of the Laplacian
    cv::Mat abs_laplacian;
    cv::convertScaleAbs(laplacian, abs_laplacian); // Convert to absolute value

    // Normalize the result to range [0, 255] for visualization
    cv::Mat normalized_abs_laplacian;
    cv::normalize(abs_laplacian, normalized_abs_laplacian, 0, 255, cv::NORM_MINMAX, CV_8U);

    //cv::imshow("Normalized Absolute Laplacian", normalized_abs_laplacian);
    //cv::waitKey(0);

    cv::Mat filtered_laplacian;
    cv::bilateralFilter(normalized_abs_laplacian, filtered_laplacian, -1, 40, 10);
    //cv::imshow("filtered laplacian", filtered_laplacian);
    //cv::waitKey(0);

    cv::Mat edges;
    cv::Canny(grad_magn, edges,150, 400);
    // cv::imshow("canny", edges);
    // cv::waitKey(0);

    cv::Mat element = cv::getStructuringElement( 
                        cv::MORPH_CROSS, cv::Size(3,3)); 

    cv::dilate(edges,edges,element,cv::Point(-1,-1),2);
    cv::erode(edges,edges,element,cv::Point(-1,-1),1);
    // cv::imshow("dilated canny", edges);
    // cv::waitKey(0);

    return edges;
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

cv::Mat contrastStretchTransform(const cv::Mat& src) {
    // Create a lookup table for faster processing
    cv::Mat lookupTable(1, 256, CV_8U);
    uchar* p = lookupTable.ptr();
    for (int i = 0; i < 256; ++i) {
        if(i < 80) {
            p[i] = cv::saturate_cast<uchar>(i/4);
        }
        else {
            p[i] = cv::saturate_cast<uchar>(i*2);
        }
        
    }

    cv::Mat dst;
    // Apply the lookup table to the source image
    cv::LUT(src, lookupTable, dst);

    return dst;
}

void addSaltPepperNoise(cv::Mat& src, cv::Mat& dst, double noise_amount) {
    dst = src.clone();
    int num_salt = noise_amount * src.total();
    int num_pepper = noise_amount * src.total();

    for (int i = 0; i < num_salt; i++) {
        int x = rand() % src.cols;
        int y = rand() % src.rows;
        dst.at<uchar>(y, x) = 255; // Pixel bianco
    }

    for (int i = 0; i < num_pepper; i++) {
        int x = rand() % src.cols;
        int y = rand() % src.rows;
        dst.at<uchar>(y, x) = 0; // Pixel nero
    }
}
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

    cv::Mat preprocessed = preprocess(image);
    
    std::vector<int> angles = {-5,-6,-7,-8,-9, -10, -11, -12, -13,-14,-15,-16};
    std::vector<float> scales = {1};
    std::vector<cv::RotatedRect> list_boxes;

    for(int l = 0; l<scales.size(); l++) {
        for(int k = 0; k<angles.size(); k++) {
        // Template size
        int template_height = 5;
        int template_width = 110*scales[l];

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

            // Rotate the template
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

            cv::Mat tm_result;
            //cv::filter2D(dilate, test, CV_32F, rotated);
            //cv::imshow("test", test);
            //cv::waitKey(0);

            // use dilate or medianblurred or canny with 100-1000
            cv::Mat tm_result_unnorm;
            cv::matchTemplate(preprocessed,rotated_template,tm_result_unnorm,cv::TM_SQDIFF,rotated_mask);
            cv::normalize( tm_result_unnorm, tm_result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );
        
            cv::imshow("TM Result", tm_result);
            cv::waitKey(0);

            // Finding local minima
            cv::Mat eroded;
            std::vector<cv::Point> minima;
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(rotated_width, rotated_height));
            cv::erode(tm_result, eroded, kernel);
            cv::Mat localMinimaMask = (tm_result == eroded) & (tm_result <= 0.05);

            // cv::imshow("TM Result, eroded", eroded);
            // cv::waitKey(0);

            // Find all non-zero points (local minima) in the mask
            cv::findNonZero(localMinimaMask, minima);

            // Draw bboxes of the found lines
            for (const cv::Point& pt : minima) {
                // Save score of local minima
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
                    cv::line(image, vertices[i], vertices[(i+1) % 4], cv::Scalar(0, 255, 0), 2);
                }
            }
        }
        cv::imshow("with lines", image);
        cv::waitKey(0);
    }

    // O uso questo come centri di k-means, o non uso NMS e quando disegno un nuovo rotatedrect controllo che il suo match sia migliore di quello precedente
    // rispetto al bbox che gli è più vicino
    // Display the image

    cv::Point2f topRightCorner(image.cols - 1, 0);
    double distanceThreshold = 300.0;
    std::vector<cv::Point2f> filtered_verts;
    // Filter the rectangles based on the distance to the top-right corner
    for (const auto& rect : list_boxes) {
        double dist =  std::sqrt(std::pow(rect.center.x - topRightCorner.x, 2) + std::pow(rect.center.y - topRightCorner.y, 2));;
        if (dist > distanceThreshold) {
            cv::Point2f vertices[4];
            rect.points(vertices);
            for (int i = 0; i < 4; i++) {
                cv::line(image, vertices[i], vertices[(i+1) % 4], cv::Scalar(0, 255, 0), 2);
                filtered_verts.push_back(vertices[i]);
            }
        }
    }

    std::vector<cv::Point2f> hull;
    cv::convexHull(filtered_verts, hull);

    std::vector<std::pair<double, std::pair<cv::Point2f, cv::Point2f>>> hullLines;    
    // Draw the convex hull
    for (size_t i = 0; i < hull.size(); i++) {
        cv::line(image, hull[i], hull[(i + 1) % hull.size()], cv::Scalar(0, 255, 0), 2);
        cv::Point2f p1 = hull[i];
        cv::Point2f p2 = hull[(i + 1) % hull.size()]; // Wrap around to form a closed hull
        double distance = cv::norm(p1-p2);
        hullLines.push_back(std::make_pair(distance, std::make_pair(p1, p2)));
    }

    // Sort the lines by their length in descending order
    std::sort(hullLines.begin(), hullLines.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });

    std::vector<double> ms;
    std::vector<double> bs;
    
    // Highlight the 4 longest lines in red
    for (size_t i = 0; i < std::min(hullLines.size(), size_t(4)); i++) {
        auto& line = hullLines[i];
        double m = static_cast<double>(line.second.second.y - line.second.first.y) / (line.second.second.x - line.second.first.x);
        double b = line.second.first.y - m * line.second.first.x;
        ms.push_back(m);
        bs.push_back(b);
        cv::line(image, line.second.first, line.second.second, cv::Scalar(0, 0, 255), (i+1)*(i+1));
    }

    std::vector<cv::Point2f> hom_points;
    // Check all pairs of lines for intersections
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

            if(x<0) x = 0;
            if(x >= image.cols) x = image.cols-1;
            if(y <0) y = 0;
            if(y >= image.rows) y = image.rows -1;

            // Check if the intersection point is inside the image
            if (x >= 0 && x < image.cols && y >= 0 && y < image.rows) {
                cv::circle(image, cv::Point(static_cast<int>(x), static_cast<int>(y)), 5, cv::Scalar(0, 0, 255), -1);
                hom_points.push_back(cv::Point2f(static_cast<int>(x), static_cast<int>(y)));
            } else {
                std::cout << "Intersection of lines " << i << " and " << j 
                          << " is at (" << x << ", " << y << ") and is outside the image." << std::endl;
            }
        }
    }

    // Iterate over the points to determine the corners
    for (const auto& point : hom_points) {
            std::cout << point << std::endl;
    }
    cv::imshow("NMS Result", image);
    cv::waitKey(0);

/*
    std::vector<cv::Point2f> to_hom_points = {cv::Point2f(999,0), cv::Point2f(999,999), cv::Point2f(0,0), cv::Point2f(0,999)};
    cv::Mat F = cv::findHomography(hom_points, to_hom_points);
    cv::Mat result(1000, 1000, CV_8U);
    cv::warpPerspective(preprocessed, result, F, cv::Size(1000,1000));
    cv::imshow("result", result);
*/
    return parkingSpots;
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

    cv::Mat gsthold;
    cv::threshold( gammaCorrected2, gsthold, 180, 255,  cv::THRESH_BINARY);
    // cv::imshow("gsthold", gsthold);
    // cv::waitKey(0);

    cv::Mat gx;
    cv::Sobel(gs, gx, CV_16S, 1,0);

    cv::Mat gy;
    cv::Sobel(gs, gy, CV_16S, 0,1);

    cv::Mat abs_grad_x;
    cv::Mat abs_grad_y;
    cv::convertScaleAbs(gx, abs_grad_x);
    cv::convertScaleAbs(gy, abs_grad_y);

    // cv::imshow("gradient x gamma", abs_grad_x);
    // cv::waitKey(0);
    // cv::imshow("gradient y gamma", abs_grad_y);
    // cv::waitKey(0);

    cv::Mat grad_magn;
    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad_magn);

    cv::Mat equalized;
    cv::equalizeHist(gammaCorrected2,equalized);
    //cv::imshow("equalized", equalized);
    //cv::waitKey(0);

    cv::Mat equalized_filt;
    cv::GaussianBlur(equalized,equalized_filt, cv::Size(5,5),30);

    cv::Mat gxeq;
    cv::Sobel(stretched, gxeq, CV_16S, 1,0);

    cv::Mat gyeq;
    cv::Sobel(stretched, gyeq, CV_16S, 0,1);

    cv::Mat abs_grad_xeq;
    cv::Mat abs_grad_yeq;
    cv::convertScaleAbs(gxeq, abs_grad_xeq);
    cv::convertScaleAbs(gyeq, abs_grad_yeq);

    // cv::imshow("gradient x eq", abs_grad_xeq);
    // cv::waitKey(0);
    // cv::imshow("gradient y eq", abs_grad_yeq);
    // cv::waitKey(0);

    //cv::Mat grad_magneq;
    //cv::addWeighted(abs_grad_xeq, 0.5, abs_grad_yeq, 0.5, 0, grad_magneq);

    cv::Mat laplacian;
    cv::Laplacian(equalized, laplacian, CV_32F);  // Use CV_32F to avoid overflow

    // Compute the absolute value of the Laplacian
    cv::Mat abs_laplacian;
    cv::convertScaleAbs(laplacian, abs_laplacian); // Convert to absolute value

    // Normalize the result to range [0, 255] for visualization
    cv::Mat normalized_abs_laplacian;
    cv::normalize(abs_laplacian, normalized_abs_laplacian, 0, 255, cv::NORM_MINMAX, CV_8U);

    //cv::imshow("Normalized Absolute Laplacian", normalized_abs_laplacian);  // Normalized for grayscale
    //cv::waitKey(0);  // Wait for a key press indefinitely

    cv::Mat filtered_laplacian;
    cv::bilateralFilter(normalized_abs_laplacian, filtered_laplacian, -1, 40, 10);
    //cv::imshow("filtered laplacian", filtered_laplacian);
    //cv::waitKey(0);

    //cv::imshow("gradient magnitude", grad_magn);
    //cv::waitKey(0);

    cv::Mat grad_magn_bilateral;
    cv::bilateralFilter(grad_magn, grad_magn_bilateral, -1, 20, 10);

    cv::Mat edges;
    cv::Canny(grad_magn, edges,100, 400);
    cv::imshow("canny", edges);
    cv::waitKey(0);

    cv::Mat medianblurred;
    cv::medianBlur(grad_magn_bilateral, medianblurred, 3);
    cv::Mat bilateralblurred;
    //cv::bilateralFilter(medianblurred, bilateralblurred, -1,20,10);
    //cv::imshow("bilateral filtered2", medianblurred);
    cv::waitKey(0);

    cv::Mat element = cv::getStructuringElement( 
                        cv::MORPH_CROSS, cv::Size(3,3)); 

    cv::dilate(edges,edges,element,cv::Point(-1,-1),2);
    cv::erode(edges,edges,element,cv::Point(-1,-1),1);
    cv::imshow("dilated canny", edges);
    cv::waitKey(0);

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
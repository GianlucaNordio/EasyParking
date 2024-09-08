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

double computeIntersectionAreaAtDifferentSize(const cv::RotatedRect& rect1, const cv::RotatedRect& rect2) {
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

    cv::Mat filteredImage;
    cv::bilateralFilter(image, filteredImage, -1, 40, 10);

    //cv::imshow("bilateral filtered", filteredImage);
    //cv::waitKey(0);

    cv::Mat gs;
    cv::cvtColor(image, gs, cv::COLOR_BGR2GRAY);

    /* cv::imshow("grayscale", gs);
    cv::waitKey(0); */

    cv::Mat stretched = contrastStretchTransform(gs);
    /* cv::imshow("stretched", stretched);
    cv::waitKey(0); */

    // Set the gamma value
    double gammaValue = 1.25; // Example gamma value

    // Apply gamma transformation
    cv::Mat gammaCorrected1 = applyGammaTransform(gs, gammaValue);
   /*  cv::imshow("gamma tf1", gammaCorrected1);
    cv::waitKey(0); */

    gammaValue = 2;
    cv::Mat gammaCorrected2 = applyGammaTransform(gammaCorrected1, gammaValue);
    /* cv::imshow("gamma tf2", gammaCorrected2);
    cv::waitKey(0);
 */
    cv::Mat gsthold;
    cv::threshold( gammaCorrected2, gsthold, 180, 255,  cv::THRESH_BINARY);
    // cv::imshow("gsthold", gsthold);
    // cv::waitKey(0);

    cv::Mat gx;
    cv::Sobel(gammaCorrected2, gx, CV_16S, 1,0);

    cv::Mat gy;
    cv::Sobel(gammaCorrected1, gy, CV_16S, 0,1);

    cv::Mat abs_grad_x;
    cv::Mat abs_grad_y;
    cv::convertScaleAbs(gx, abs_grad_x);
    cv::convertScaleAbs(gy, abs_grad_y);

    /* cv::imshow("gradient x gamma", abs_grad_x);
    cv::waitKey(0);
    cv::imshow("gradient y gamma", abs_grad_y);
    cv::waitKey(0); */

    cv::Mat grad_magn;
    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad_magn);

    cv::Mat equalized;
    cv::equalizeHist(gammaCorrected2,equalized);
   /*  cv::imshow("equalized", equalized);
    cv::waitKey(0); */

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

    /* cv::imshow("Normalized Absolute Laplacian", normalized_abs_laplacian);  // Normalized for grayscale
    cv::waitKey(0); */  // Wait for a key press indefinitely

    cv::Mat filtered_laplacian;
    cv::bilateralFilter(normalized_abs_laplacian, filtered_laplacian, -1, 40, 10);
    /* cv::imshow("filtered laplacian", filtered_laplacian);
    cv::waitKey(0); */

    /* cv::imshow("gradient magnitude", grad_magn+normalized_abs_laplacian);
    cv::waitKey(0); */

    cv::Mat grad_magn_bilateral;
    cv::bilateralFilter(grad_magn, grad_magn_bilateral, -1, 20, 10);
   /*  cv::imshow("grad magn bilateral", grad_magn_bilateral);
    cv::waitKey(0);
 */
    cv::Mat medianblurred;
    cv::medianBlur(grad_magn_bilateral, medianblurred, 3);
    cv::Mat bilateralblurred;
    //cv::bilateralFilter(medianblurred, bilateralblurred, -1,20,10);
    /* cv::imshow("bilateral filtered2", medianblurred);
    cv::waitKey(0); */

    cv::Mat element = cv::getStructuringElement( 
                        cv::MORPH_RECT, cv::Size(3,3)); 

    cv::Mat erodeg; 
    cv::erode(medianblurred, erodeg, element, cv::Point(-1, -1), 1); 
/*     cv::imshow("erodeg", erodeg);
 */
    //cv::Mat gmagthold;
    //cv::threshold( erodeg, gmagthold, 110, 255,  cv::THRESH_BINARY);
    //cv::imshow("gmagthold", gmagthold);
    //cv::waitKey(0);

    cv::Mat dilate = grad_magn+normalized_abs_laplacian; 
    // ok with 5-6 dilations for normal and flipped. Good with 8 for normal
    cv::Mat dilate_pre_filter;
    cv::dilate(grad_magn, dilate_pre_filter, element, cv::Point(-1, -1), 3);
    cv::bilateralFilter(dilate_pre_filter, dilate, -1, 20, 10);
    // dilate = grad_magn;
    //dilate = applyGammaTransform(dilate,1.2);
    dilate = dilate+erodeg;
    dilate.setTo(0, dilate<= 20);
/*     cv::imshow("dilated", dilate);
 */
/*
   std::vector<int> angles = {-8, -9, -10, -11, -12, -13};
    std::vector<float> scales = {0.75, 1, 1.05, 1.1, 1.2, 2};
    std::vector<cv::RotatedRect> list_boxes;

    for(int k = 0; k<angles.size(); k++) {
        // Template size
        int template_height = 5*2+20*scale*scale;
        int template_width = 100*scale;

        // Horizontal template and mask definition
        cv::Mat horizontal_template(template_height,template_width,CV_8U);
        cv::Mat horizontal_mask(template_height,template_width,CV_8U);

        // Build the template and mask
        for(int i = 0; i< horizontal_template.rows; i++) {
            for(int j = 0; j<horizontal_template.cols; j++) {
                uchar val = 0;
                if(i < 5 || (i > template_height-5) || j > template_width-5) {
                    val = 255;
                }
                horizontal_mask.at<uchar>(i,j) = val;
                horizontal_template.at<uchar>(i,j) = val;
            }
        }
*/
    std::vector<float> angles = {-5, -5.5, -6, -6.5, -7, -7.5, -8, -8.5, -9, -9.5, -10, -10.5, -11, -11.5, -12, -12.5, -13, -13.5, -14, -14.5, -15, -15.5, -16};
    std::vector<float> scales = {0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2};
    std::vector<cv::RotatedRect> boxes_best_angle;
    std::vector<std::pair<cv::RotatedRect, double>> final_boxes;
    for(int l = 0; l<scales.size(); l++) {
        std::vector<std::pair<cv::RotatedRect, double>> list_boxes;
        for(int k = 0; k<angles.size(); k++) {
            // Template size
            int surplus = 30*scales[k];
            int line_width = 8;
            int template_height = 39*scales[l];
            int template_width = 120*scales[l]+2*surplus;

            // Horizontal template and mask definition
            cv::Mat horizontal_template(template_height,template_width,CV_8U,cv::Scalar(0));
            cv::Mat horizontal_mask(template_height,template_width,CV_8U,cv::Scalar(0));
/*
            for(int i = 0; i< horizontal_template.rows; i++) {
                for(int j = 0; j<horizontal_template.cols; j++) {
                    if((i<line_width && (j > surplus && j<template_width-surplus)) 
                        || (j > surplus && j < surplus+line_width)
                        || (j > template_width-line_width-surplus && j<template_width-surplus) 
                        || (i > (template_height-line_width) && (j > surplus && j<template_width-surplus))){
                        horizontal_template.at<uchar>(i,j) = 220;
                        horizontal_mask.at<uchar>(i,j) = 255;
                    }
                    else {
                        horizontal_mask.at<uchar>(i,j) = 127;
                    }
                }
            }
*/
            // Build the template and mask
            for(int i = 0; i< horizontal_template.rows; i++) {
                for(int j = 0; j<horizontal_template.cols; j++) {
                    if(((i<line_width && j > surplus) 
                        || (j > template_width-line_width/2) 
                        || (i > (template_height-line_width) && j > (20*scales[l]*scales[l]+surplus/2)))){
                        horizontal_template.at<uchar>(i,j) = 220;
                        horizontal_mask.at<uchar>(i,j) = 200;
                    }
                    else {
                        horizontal_mask.at<uchar>(i,j) = 100;
                    }
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

            float rotated_width = template_width*cos(-angles[k]*CV_PI/180)+line_width;
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
            cv::matchTemplate(dilate,rotated_template,tm_result_unnorm,cv::TM_CCORR_NORMED,rotated_mask);
            cv::normalize( tm_result_unnorm, tm_result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );
        
            // cv::imshow("TM Result", tm_result);
            // cv::waitKey(0);

            // Finding local minima
            cv::Mat eroded;
            std::vector<cv::Point> minima;
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(rotated_width, rotated_height));
            cv::dilate(tm_result, eroded, kernel);
            cv::Mat localMinimaMask = (tm_result == eroded) & (tm_result >= 0.975);

            // cv::imshow("TM Result, eroded", eroded);
            // cv::waitKey(0);

            // Find all non-zero points (local minima) in the mask
            cv::findNonZero(localMinimaMask, minima);

            // Draw bboxes of the found lines
            for (const cv::Point& pt : minima) {
                // Save score of local minima

                // White circles at minima points
                // cv::circle(gs, pt, 3, cv::Scalar(255), 1);

                // Get center of the bbox to draw the rotated rect
                cv::Point center;
                center.x = pt.x+rotated_width/2;
                center.y = pt.y+rotated_height/2;

                cv::RotatedRect rotatedRect(center, cv::Size(template_width-2*surplus,template_height), -angles[k]);
                list_boxes.push_back(std::pair(rotatedRect, tm_result.at<double>(pt)));

                // Draw the rotated rectangle using lines between its vertices
                // cv::Point2f vertices[4];
                // rotatedRect.points(vertices);
                // for (int i = 0; i < 4; i++) {
                //     cv::line(image, vertices[i], vertices[(i+1) % 4], cv::Scalar(0, 255, 0), 2);
                // }
            }
        }

        std::cout << "Number of boxes: " << list_boxes.size() << std::endl;

        std::vector<std::pair<cv::RotatedRect, double>> elementsToRemove;

        for (const auto& box : list_boxes) {
            for (const auto& box2 : list_boxes) {
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
            std::vector<std::pair<cv::RotatedRect, double>>::const_iterator iterator = elementIterator(list_boxes, element);
            if (iterator != list_boxes.cend()) {
                list_boxes.erase(iterator);
            }
        }
        
        float scoreThreshold = 0.0f;  // Minimum score to keep
        float nmsThreshold = 0.5f;    // IoU threshold for NMS
        // Vector to store indices of bounding boxes to keep after NMS
        std::vector<int> indices;
        std::vector<float> scores(list_boxes.size(),0.1);

        std::vector<cv::RotatedRect> list_boxes_onlyRect;
        for(auto box : list_boxes) {
            final_boxes.push_back(box);
            list_boxes_onlyRect.push_back(box.first);
        }
/* 
        for(auto rect : list_boxes_onlyRect) {
            // Draw the rotated rectangle
            cv::Point2f vertices[4];
            rect.points(vertices);
            for (int j = 0; j < 4; j++) {
                cv::line(image, vertices[j], vertices[(j + 1) % 4], cv::Scalar(0, 0, 255), 2);
            }
            
        } */
        /* // Apply NMS for rotated rectangles
        cv::dnn::NMSBoxes(list_boxes_onlyRect, scores, scoreThreshold, nmsThreshold, indices);
        
        // Draw the remaining boxes after NMS
        for (int idx : indices) {
            cv::RotatedRect& rect = list_boxes_onlyRect[idx];
            boxes_best_angle.push_back(rect);
            // Draw the rotated rectangle
            cv::Point2f vertices[4];
            rect.points(vertices);
            for (int j = 0; j < 4; j++) {
                cv::line(image, vertices[j], vertices[(j + 1) % 4], cv::Scalar(0, 0, 255), 2);
            }
        } */
        /* cv::imshow("with lines", image);
        cv::waitKey(0); */
    }

    float scoreThold = 0.0f;
    float nmsThold = 0.5f;
    std::vector<int> indices_final;
    std::vector<float> scores_final(boxes_best_angle.size(),0.1);
    // Apply NMS for rotated rectangles
    /* cv::dnn::NMSBoxes(boxes_best_angle, scores_final, scoreThold, nmsThold, indices_final);
        // Draw the remaining boxes after NMS
        for (int idx : indices_final) {
            cv::RotatedRect& rect = boxes_best_angle[idx];
            // Draw the rotated rectangle
            cv::Point2f vertices[4];
            rect.points(vertices);
            for (int j = 0; j < 4; j++) {
                cv::line(image, vertices[j], vertices[(j + 1) % 4], cv::Scalar(255, 0, 0), 2);
            }
        }
    // O uso questo come centri di k-means, o non uso NMS e quando disegno un nuovo rotatedrect controllo che il suo match sia migliore di quello precedente
    // rispetto al bbox che gli è più vicino
    // Display the image */
    
    std::vector<std::pair<cv::RotatedRect, double>> elementsToRemove;

    for(const auto& box : final_boxes) {
            for (const auto& box2 : final_boxes) {
                //Check if box and box2 are in the same position on list_boxes, so are the same rect)
                

                cv::RotatedRect rect1 = box.first;
                cv::RotatedRect rect2 = box2.first;


                double score1 = box.second;
                double score2 = box2.second;

                if(rect1.center.x == rect2.center.x && rect1.center.y == rect2.center.y && score1 == score2) {
                    std::cout << "same rect" << std::endl;
                } else if(computeIntersectionAreaAtDifferentSize(rect1,rect2) > 0.4) {
                    if( rect1.size.area() >= rect2.size.area()) {
                        elementsToRemove.push_back(box2);
                    }
                    else {
                        elementsToRemove.push_back(box);
                    }
                }
            }
        }

    for (std::pair element : elementsToRemove) {
            std::vector<std::pair<cv::RotatedRect, double>>::const_iterator iterator = elementIterator(final_boxes, element);
            if (iterator != final_boxes.cend()) {
                final_boxes.erase(iterator);
            }
        }

    std::vector<cv::RotatedRect> list_boxes_onlyRect;
    for(auto box : final_boxes) {
        list_boxes_onlyRect.push_back(box.first);
    }

    for(auto rect : list_boxes_onlyRect) {
        // Draw the rotated rectangle
        cv::Point2f vertices[4];
        rect.points(vertices);
        for (int j = 0; j < 4; j++) {
            cv::line(image, vertices[j], vertices[(j + 1) % 4], cv::Scalar(0, 255, 0), 2);
        }
        
    }

    cv::imshow("NMS Result", image);
    cv::waitKey(0);

    return parkingSpots;
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
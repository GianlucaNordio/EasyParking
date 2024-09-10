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

    cv::Mat gaussianFilteredImage;
    cv::GaussianBlur(image,gaussianFilteredImage,cv::Size(3,3),20);
    cv::Mat filteredImage;
    cv::bilateralFilter(image, filteredImage, -1, 40, 10);

    cv::Mat gs;
    cv::cvtColor(filteredImage, gs, cv::COLOR_BGR2GRAY);

    cv::Mat adpt;
    cv::adaptiveThreshold(gs,adpt,255, cv::ADAPTIVE_THRESH_MEAN_C ,cv::THRESH_BINARY, 9,-20);
    
    cv::Mat element = cv::getStructuringElement( 
                        cv::MORPH_CROSS, cv::Size(3,3)); 

    // dil 2 erode 1
    cv::dilate(adpt,adpt,element,cv::Point(-1,-1),4);
    cv::erode(adpt,adpt,element,cv::Point(-1,-1),3);

    cv::imshow("grayscale", adpt);
    cv::waitKey(0);
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
    std::vector<float> angles = {-5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16};
    std::vector<float> scales = {0.5, 1, 1.5};
    std::vector<cv::RotatedRect> boxes_best_angle;
    std::vector<std::pair<cv::RotatedRect, double>> final_boxes;
    for(int l = 0; l<scales.size(); l++) {
        std::vector<std::pair<cv::RotatedRect, double>> list_boxes;
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
            cv::matchTemplate(adpt,rotated_template,tm_result_unnorm,cv::TM_SQDIFF,rotated_mask);
            cv::normalize( tm_result_unnorm, tm_result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );
        
            cv::imshow("TM Result", tm_result);
            cv::waitKey(0);

            // Finding local minima
            cv::Mat eroded;
            std::vector<cv::Point> minima;
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(rotated_width, rotated_height));
            cv::erode(tm_result, eroded, kernel);
            cv::Mat localMinimaMask = (tm_result == eroded) & (tm_result < 0.2);

            cv::imshow("TM Result, eroded", eroded);
            cv::waitKey(0);

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

                cv::RotatedRect rotatedRect(center, cv::Size(template_width,template_height), -angles[k]);
                list_boxes.push_back(std::pair(rotatedRect, tm_result.at<double>(pt)));

                // Draw the rotated rectangle using lines between its vertices
                cv::Point2f vertices[4];
                rotatedRect.points(vertices);
                for (int i = 0; i < 4; i++) {
                    cv::line(image, vertices[i], vertices[(i+1) % 4], cv::Scalar(0, 255, 0), 2);
                }
            }
            cv::imshow("tm result image", image);
            cv::waitKey(0);
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
                    if (score1 <= score2) {
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
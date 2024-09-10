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
    cv::Mat preprocessed = preprocess(image);
    cv::Mat intermediate_results = image.clone();
    
	cv::Ptr<cv::LineSegmentDetector > lsd = cv::createLineSegmentDetector();
    std::vector<cv::Vec4f> line_segm;
    lsd->detect(preprocessed,line_segm);
    std::vector<cv::Vec4f> segments;

    // Draw the line segments on the image
    for (const auto& segment : line_segm) {
        cv::line(intermediate_results, cv::Point(segment[0], segment[1]), cv::Point(segment[2], segment[3]), 
                 cv::Scalar(0, 0, 255), 2, cv::LINE_AA);  // Red color lines with thickness 2
    }

    // Display the result
    cv::imshow("Detected Line Segments", intermediate_results);
    cv::waitKey(0);

	//rejecting short segments
    for(int i = 0; i<line_segm.size(); i++)
    {                                                                                                                         //150
        if((line_segm[i][0] - line_segm[i][2])*(line_segm[i][0] - line_segm[i][2]) + (line_segm[i][1] - line_segm[i][3])*(line_segm[i][1] - line_segm[i][3]) > 500)
        {
            segments.push_back(line_segm[i]);
        }
    }
	/*
	Mat copied = img.clone();
	lsd->drawSegments(copied,segments);
	imshow("segm",copied);
	waitKey(0);
	*/
	//TODO: PROVARE A NON USARE GLI IF BLUESTART CON OFFEST%4
	for(int i = 1; i<segments.size(); i++)
    {
		cv::Vec2f mq_param = get_segm_params(segments[i]);
		cv::Point2f blue;
		cv::Point2f red;
		cv::Point2f green;
		cv::Point2f yellow;
		
		if(segments[i][0]<segments[i][2])//BLUE_START
		{
			blue = cv::Point2f(segments[i][0],segments[i][1]);
			red = cv::Point2f(segments[i][2],segments[i][3]);
			
			cv::Vec2f doble = get_direction(segments[i],true);//vector from blue to red
			
			green = cv::Point2f(segments[i][2]+doble[0],segments[i][3]+doble[1]);//red + doble_vector
			
			if(mq_param[0]>0)
			{
				//drawMarker(copied,Point2f(segments[i][2]+doble[0]*0.7    - doble[1],segments[i][3]+doble[1]+doble[0]*0.7   ),Scalar(0,255,255),MARKER_TILTED_CROSS,7,3);
				yellow = cv::Point2f(segments[i][2]+doble[0]    - doble[1]*0.7,segments[i][3]+doble[1]+doble[0]*0.7   );
			}
			else
			{
				//drawMarker(copied,Point2f(segments[i][2]+doble[0] + doble[1]*2   ,segments[i][3]+doble[1]-doble[0]*2   ),Scalar(0,255,255),MARKER_TILTED_CROSS,7,3);
				yellow = cv::Point2f(segments[i][2]+doble[0] + doble[1]*1.7   ,segments[i][3]+doble[1]-doble[0]*1.7   );
			}
		}
		else//RED_START
		{
			red = cv::Point2f(segments[i][0],segments[i][1]);
			blue = cv::Point2f(segments[i][2],segments[i][3]);

			cv::Vec2f doble = get_direction(segments[i],false);

			green = cv::Point2f(segments[i][0]+doble[0],segments[i][1]+doble[1]);//red + doble_vector

			if(mq_param[0]>0)
			{
				yellow = cv::Point2f(segments[i][0]+doble[0] - doble[1]*0.7   ,segments[i][1]+doble[1]+doble[0]*0.7   );
			}
			else
			{
				yellow = cv::Point2f(segments[i][0]+doble[0] + doble[1]*1.7   ,segments[i][1]+doble[1]-doble[0]*1.7   );
			}
		}
		
			/*
			Mat another = img.clone();
			drawMarker(img,red,Scalar(0,0,255),MARKER_TILTED_CROSS,15,3);
			drawMarker(img,blue,Scalar(255,255,0),MARKER_TILTED_CROSS,15,3);
			drawMarker(img,green,Scalar(0,255,0),MARKER_TILTED_CROSS,15,3);
			drawMarker(img,yellow,Scalar(0,255,255),MARKER_TILTED_CROSS,15,3);
			imshow("another",another);
			waitKey(0);
			*/
		
		cv::RotatedRect sp = cv::RotatedRect(blue,green,yellow);
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

    // Display the result
    cv::imshow("Rotated Rectangles", image);
    cv::waitKey(0);
	
	return parkingSpots;
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

    cv::Mat adpt;
    cv::adaptiveThreshold(gs,adpt,255, cv::ADAPTIVE_THRESH_MEAN_C ,cv::THRESH_BINARY, 9,-20);
    
    cv::Mat element = cv::getStructuringElement( 
                        cv::MORPH_CROSS, cv::Size(3,3)); 

    // dil 2 erode 1
    cv::dilate(adpt,adpt,element,cv::Point(-1,-1),4);
    cv::erode(adpt,adpt,element,cv::Point(-1,-1),3);

    cv::imshow("grayscale", adpt);
    cv::waitKey(0);
    return adpt;

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

    //cv::Mat element = cv::getStructuringElement( 
    //                    cv::MORPH_CROSS, cv::Size(3,3)); 

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
            if (!(rect1.center.x == rect2.center.x && rect1.center.y == rect2.center.y) && (computeIntersectionArea(rect1, rect2) > 0.4)) {
                if (rect1.size.area() > rect2.size.area()){
                    elementsToRemove.push_back(rect2);
                } else {
                    elementsToRemove.push_back(rect1);
                }
            }
        }
    }
}
#include "minimap.hpp"

/**
 * @brief Builds a series of minimaps for a sequence of parking spot data.
 * 
 * This function iterates over multiple sets of parking spot data and generates a minimap for each set.
 * It calls `buildMinimap` for each set of parking spots to draw the corresponding minimap on the provided
 * images in the `miniMaps` vector.
 * 
 * @param parkingSpots A vector of vectors, where each inner vector contains `ParkingSpot` objects representing
 * the parking spots for a specific image.
 * @param miniMaps A vector of `cv::Mat` objects where each element will be updated with the minimap corresponding
 * to the parking spots in the `parkingSpots` vector. Each `cv::Mat` should be pre-allocated with appropriate size
 * and type.
 * 
 * @note The size of the `miniMaps` vector must match the size of the `parkingSpots` vector, as each minimap 
 * corresponds to one set of parking spots.
 * 
 * @throws std::out_of_range If the `miniMaps` vector does not have enough elements to match the number of 
 * `parkingSpots` vectors.
 */
void buildSequenceMinimap(std::vector<std::vector<ParkingSpot>> parkingSpots, std::vector<cv::Mat>& miniMaps) {
    for(int i = 0; i < parkingSpots.size(); i++) {
        buildMinimap(parkingSpots[i], miniMaps[i]);
    }
}

/**
 * @brief Builds a minimap representing parking spots and their occupancy status on a convex hull.
 * 
 * This function creates a minimap image based on the parking spot data provided. It first computes
 * the convex hull of the parking spots' bounding boxes and highlights the four longest edges. The function
 * then calculates the intersection points of the lines to determine the corners, applies a perspective
 * transformation, and draws the transformed parking spots on the minimap.
 * 
 * Each parking spot is displayed with a color representing its occupancy status (red for occupied, blue 
 * for free), and the bounding box of the parking spot is drawn at a transformed location in the minimap.
 * 
 * @param parkingSpot A vector of `ParkingSpot` objects representing the parking spots to be displayed 
 * on the minimap. Each `ParkingSpot` contains a `cv::RotatedRect` for its bounding box and an occupancy 
 * flag.
 * @param miniMap A reference to a `cv::Mat` object where the minimap will be drawn. This matrix is 
 * expected to have the appropriate size and type for rendering the minimap.
 * 
 * @note The function assumes that all parking spots in the input have valid, non-zero area bounding boxes. 
 * Parking spots with an area smaller than 1 are ignored. It is also assumed that the `miniMap` has been 
 * initialized to a blank image of appropriate size.
 * 
 * @throws std::invalid_argument If the `parkingSpot` vector is empty.
 */
void buildMinimap(std::vector<ParkingSpot> parkingSpot, cv::Mat& miniMap) {
    std::vector<cv::Point2f> allVertices;
    
    for (ParkingSpot& spot : parkingSpot) {
        cv::RotatedRect rect = spot.rect;
        // The if is needed because removing with the iterator produces rects with zero area
        if(rect.size.area()>1) {
            cv::Point2f vertices[4];
            rect.points(vertices);

            for (int i = 0; i < 4; i++) {
                allVertices.push_back(vertices[i]);
            }
        }
    }

    std::vector<cv::Point2f> hull;
    cv::convexHull(allVertices, hull);

    // Draw the convex hull
    std::vector<std::pair<double, cv::Vec4f>> hullLines;    
    for (size_t i = 0; i < hull.size(); i++) {
        cv::Point2f p1 = hull[i];
        cv::Point2f p2 = hull[(i + 1) % hull.size()];
        double distance = cv::norm(p1-p2);
        hullLines.push_back(std::make_pair(distance, cv::Vec4f(p1.x, p1.y, p2.x, p2.y)));

    }

    // Sort the lines by their length in descending order
    std::sort(hullLines.begin(), hullLines.end(), [](const auto& a, const auto& b) { 
        return a.first > b.first;
    });

    std::vector<double> slopes;
    std::vector<double> intercepts;

    // Get the equations (slope and intercept) of the four longest lines
    for (size_t i = 0; i < std::min(hullLines.size(), size_t(4)); i++) {
        std::pair<double, cv::Vec4f>& line = hullLines[i];
        double slope = tan(getSegmentAngularCoefficient(line.second)*CV_PI/180);
        double intercept = line.second[1] - slope * line.second[0];

        slopes.push_back(slope);
        intercepts.push_back(intercept);
    }

    std::vector<cv::Point2f> hullCorners;

    // Check all pairs of lines for intersections
    for (size_t i = 0; i < slopes.size(); ++i) {
        for (size_t j = i + 1; j < slopes.size(); ++j) {

            double slope1 = slopes[i];
            double intercept1 = intercepts[i];
            double slope2 = slopes[j];
            double intercept2 = intercepts[j];

            // If lines have the same sign of slope, then we don't need to take their intersection
            if ((slope1 < 0) == (slope2 < 0)) {
                continue;
            }

            double x = (intercept2 - intercept1) / (slope1 - slope2);
            double y = slope1 * x + intercept1;
            hullCorners.push_back(cv::Point2f(x, y));
        }
    }

    // Sort the corner points
    std::vector<cv::Point2f> hullCornersSorted = findCorners(hullCorners);
    cv::Size mapSize(MAP_WIDTH, MAP_HEIGHT);
    
    std::vector<cv::Point2f> toHomPoints = {cv::Point2f(0,MAP_HEIGHT-1), cv::Point2f(0,-25), cv::Point2f(MAP_WIDTH-1,MAP_HEIGHT-1), cv::Point2f(MAP_WIDTH-1,-25)};
    cv::Mat perspectiveTransform = cv::getPerspectiveTransform(hullCornersSorted, toHomPoints);
    cv::Mat minimap(mapSize, IMAGE_TYPE_3_CANALI, WHITE);

    double sumAngle = 0;
    double avgAngle;

    std::vector<cv::RotatedRect> transformedRects;
    std::vector<bool> occupancies;

    for(ParkingSpot spot: parkingSpot) {

        // Extract the vertices of the current RotatedRect
        cv::RotatedRect rect = spot.rect;
        occupancies.push_back(spot.occupied);
        cv::Point2f vertices[4];
        rect.points(vertices);

        // Prepare vectors to hold the original and transformed vertices
        std::vector<cv::Point2f> toTransform(vertices, vertices + 4);
        std::vector<cv::Point2f> transformedVertices;

        // Apply perspective transformation
        cv::perspectiveTransform(toTransform, transformedVertices, perspectiveTransform);

        // Compute the minimum area rectangle from the transformed vertices
        cv::RotatedRect minRect = cv::minAreaRect(transformedVertices);
        transformedRects.push_back(minRect);
        sumAngle += minRect.angle;

    }
    
    avgAngle = sumAngle/parkingSpot.size();
    alignRects(transformedRects, ALIGNED_RECTS_THRESHOLD);

    double offsetY = (miniMap.rows-minimap.rows)/2;
    double offsetX  = (miniMap.cols-minimap.cols)/2;

    for(int i = 0; i<transformedRects.size(); i++) {

        cv::RotatedRect rect = transformedRects[i];
        bool occupancy = occupancies[i];
        cv::RotatedRect toPrint(cv::Point2f(rect.center.x+offsetX, rect.center.y+offsetY), SIZE_RECT_MINIMAP,(rect.size.aspectRatio()> 1.4 ? avgAngle : -avgAngle));
        cv::Point2f vertices[4];
        toPrint.points(vertices);

        for (int i = 0; i < 4; i++) {
            // Draw the bounding box red if the parking spot is occupied, blue otherwise
            cv::line(miniMap, vertices[i], vertices[(i + 1) % 4], occupancy ? RED : BLUE, LINE_THICKNESS);
        }
    }
}

/**
 * @brief Finds and arranges the corners of a quadrilateral from a set of 4 points.
 * 
 * This function takes a vector of 4 points and arranges them into a specific order to represent the
 * corners of a quadrilateral. The function sorts the points by their y-coordinate to determine which
 * points are the top and bottom ones. Then, it further sorts the top and bottom points by their x-coordinate
 * to determine the left and right corners.
 * 
 * @param points A vector of cv::Point2f objects containing exactly 4 points. These points are expected to
 * form the corners of a quadrilateral.
 * 
 * @return A vector of cv::Point2f objects representing the corners of the quadrilateral in the following
 * order: top-left, top-right, bottom-left, and bottom-right.
 * 
 * @throws std::invalid_argument If the input vector does not contain exactly 4 points.
 * 
 * @note The input points are assumed to form a convex quadrilateral. The function will not handle cases where
 * the points are not in such an arrangement.
 */
std::vector<cv::Point2f> findCorners(const std::vector<cv::Point2f>& points) {
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

/**
 * @brief Aligns rectangles along their y-coordinates based on a specified threshold.
 * 
 * This function adjusts the y-coordinates of a set of rotated rectangles so that rectangles within
 * a given threshold of each other are aligned to the same y-coordinate. The rectangles are first
 * sorted by their y-coordinate, and then rectangles with y-coordinates close to each other are aligned
 * to the base y-coordinate of the first rectangle in their group.
 * 
 * @param rects A vector of cv::RotatedRect objects representing the rectangles to be aligned.
 * @param threshold A double value representing the maximum allowable distance between y-coordinates 
 * for rectangles to be considered for alignment. Rectangles whose y-coordinates differ by this amount 
 * or less will be aligned to the same y-coordinate.
 * 
 * @note The function modifies the y-coordinates of the rectangles in the input vector in-place.
 */
void alignRects(std::vector<cv::RotatedRect>& rects, double threshold) {
    
    // Sort the rects by their y-coordinate
    std::sort(rects.begin(), rects.end(), 
        [](const cv::RotatedRect& a, const cv::RotatedRect& b) {
            return a.center.y < b.center.y;
        });

    // Iterate through each unique y-coordinate and align rects within the threshold
    for (size_t i = 0; i < rects.size(); ++i) {
        double base_y = rects[i].center.y;

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

/**
 * @brief Adds a minimap to a sequence of images by overlaying it in a defined region.
 * 
 * This function overlays minimaps onto a sequence of images. The minimaps are placed in the bottom-left 
 * corner of each corresponding image in the sequence. The function ensures that the minimap dimensions 
 * fit within the respective images before copying.
 * 
 * @param miniMap A vector of cv::Mat objects representing the minimaps to be added. Each minimap 
 * corresponds to an image in the sequence.
 * @param sequence A vector of cv::Mat objects representing the sequence of images onto which the minimaps 
 * will be overlaid. The minimap is copied into the bottom-left region of each image.
 * 
 * @note It is assumed that the size of the miniMap vector is equal to the size of the sequence vector. 
 * If a minimap is larger than its corresponding image in either dimension, it will not be added.
 */

void addMinimap(const std::vector<cv::Mat>& miniMap, std::vector<cv::Mat>& sequence) {
    for (size_t i = 0; i < sequence.size(); i++) {
        // Verify that the minimap fits inside the sequence image
        if (miniMap[i].rows <= sequence[i].rows && miniMap[i].cols <= sequence[i].cols) {           
            // Define the region of interest (ROI) in the sequence image
            cv::Mat roi = sequence[i](cv::Rect(0, sequence[i].rows - miniMap[i].rows, miniMap[i].cols, miniMap[i].rows));

            // Copy the minimap into the region of interest previously created
            miniMap[i].copyTo(roi);
        }
    }
}
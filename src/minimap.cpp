#include "minimap.hpp"
void buildSequenceMinimap(std::vector<std::vector<ParkingSpot>> parkingSpots, std::vector<cv::Mat>& miniMaps) {
    for(int i = 0; i < parkingSpots.size(); i++) {
        buildMinimap(parkingSpots[i], miniMaps[i]);
    }
}



void buildMinimap(std::vector<ParkingSpot> parkingSpot, cv::Mat& miniMap) {

    std::vector<cv::Point2f> all_vertices;

    for (ParkingSpot& spot : parkingSpot) {
        cv::RotatedRect rect = spot.rect;
        // The if is needed because removing with the iterator produces rects with zero area
        if(rect.size.area()>1) { 

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
        cv::Point2f p2 = hull[(i + 1) % hull.size()];
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


    for(ParkingSpot spot: parkingSpot) {

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

        sum_angle += minrect.angle;

    }

    
    avg_angle = sum_angle/parkingSpot.size();

    align_rects(transformed_rects,30);

    

    // Example larger image (e.g., 800x800 white background)


    double offset_y = (miniMap.rows-minimap.rows)/2;

    double offset_x  = (miniMap.cols-minimap.cols)/2;

    for(int i = 0; i<transformed_rects.size(); i++) {

        cv::RotatedRect rect = transformed_rects[i];

        bool occupancy = occupancies[i];

        cv::RotatedRect to_print(cv::Point2f(rect.center.x+offset_x,rect.center.y+offset_y),cv::Size(60,20),(rect.size.aspectRatio()>1.4?avg_angle:-avg_angle));

        cv::Point2f vertices[4];

        to_print.points(vertices);

        for (int i = 0; i < 4; i++) {

            cv::line(miniMap, vertices[i], vertices[(i+1) % 4], occupancy ? cv::Scalar(0, 0, 255) : cv::Scalar(255,0,0), 2);

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

void addMinimap(std::vector<cv::Mat>& minimap, const std::vector<cv::Mat>& sequence) {
    for (size_t i = 0; i < sequence.size(); ++i) {
        // Assicurati che le dimensioni del minimap siano più piccole rispetto alla sequenza
        if (minimap[i].rows <= sequence[i].rows && minimap[i].cols <= sequence[i].cols) {
            // Definire la regione in cui sovrastampare il minimap (in basso a sinistra)
            cv::Mat roi = sequence[i](cv::Rect(0, sequence[i].rows - minimap[i].rows, 
                                               minimap[i].cols, minimap[i].rows));

            // Copia il minimap sull'immagine di destinazione
            minimap[i].copyTo(roi);
        }
    }
}

#include <iostream>
#include <fstream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp> 
#include "parkingSpot.hpp"

// This method reads token by token and when it finds a specific token it knows it will see a number after it
std::vector<ParkingSpot> parseXML(const std::string& filename) {
    std::vector<ParkingSpot> spaces;
    std::ifstream file(filename);
    std::string line;
    ParkingSpot currentSpace;

    std::getline(file, line); // to skip the first line containing parking (the first id would cause problems)

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        while(iss >> token) {
            if(token.find("<space") != std::string::npos) {
                currentSpace = ParkingSpot();
            } else if(token.find("</space>") != std::string::npos) {
                spaces.push_back(currentSpace);
            } else if(token.find("id=") != std::string::npos) {
                currentSpace.id = std::stoi(token.substr(token.find("\"") + 1, token.find_last_of("\"") - token.find("\"") - 1));
            } else if (token.find("occupied=") != std::string::npos) {
                currentSpace.occupied = std::stoi(token.substr(token.find("occupied=\"") + 10, 1));
            } else if (token.find("center") != std::string::npos) {
                // Read the center position
                iss >> token;
                int center_x = std::stoi(token.substr(token.find("=") + 2));
                iss >> token;
                int center_y = std::stoi(token.substr(token.find("=") + 2));
                //Add center position
                currentSpace.rect.center = cv::Point2f(center_x, center_y);
            } else if (token.find("size") != std::string::npos) {
                //Read the rectangle size
                iss >> token;
                int width = std::stoi(token.substr(token.find("=") + 2));
                iss >> token;
                int height = std::stoi(token.substr(token.find("=") + 2));
                // Add rectangle size
                currentSpace.rect.size = cv::Size2f(width, height);
            } else if (token.find("angle") != std::string::npos) {
                iss >> token;
                currentSpace.rect.angle = std::stoi(token.substr(token.find("=") + 2));
            }
        }
        
        /*
        else if (token.find("point") != std::string::npos) {
                Point point;
                iss >> token;
                point.x = std::stoi(token.substr(token.find("=") + 2));
                iss >> token;
                point.y = std::stoi(token.substr(token.find("=") + 2));
                currentSpace.contour.push_back(point);
            }
        */
        
    }
    return spaces;
}


/* // Obtain a vector of ParkingSpots from parseXML
    std::vector<ParkingSpot> spaces = parseXML("../src/test_bbox.xml");
    std::cout<< spaces.size();
    for (const auto& space : spaces) {
        std::cout << "Space ID: " << space.id << "\n";
        std::cout << "Occupied: " << space.occupied << "\n";
        std::cout << "Center: (" << space.rect.center.x << "," <<space.rect.center.y << ")\n";
        std::cout << "Size: " << space.rect.size.width << "x" << space.rect.size.height << "\n";
        std::cout << "Angle: " << space.rect.angle << "\n";
        //std::cout << "Contour points: ";
        //for (const auto& point : space.contour) {
        //    std::cout << "(" << point.x << ", " << point.y << ") ";
        //}
        std::cout << "\n\n";

        // Adding rectangles to the image
        cv::Point2f vertices[4];
        space.rect.points(vertices);
        for (int i = 0; i < 4; i++) {
            cv::line(img, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2); 
        }
    }
*/
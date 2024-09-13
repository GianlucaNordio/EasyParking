#include <iostream>
#include <fstream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp> 
#include "parkingSpot.hpp"

void parseXML(const std::string& filePath, std::vector<ParkingSpot> &spaces) {
    std::ifstream file(filePath);
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
    }
}
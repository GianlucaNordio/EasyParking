#ifndef PARSER_HPP
#define PARSER_HPP

#include <opencv2/opencv.hpp>

/**
 * @brief reads the parking spots from an XML file 
 * @param filePath path of an XML file to parse
 */
void parseXML(const std::string& filePath, std::vector<ParkingSpot> &spaces);

#endif // PARSER_HP
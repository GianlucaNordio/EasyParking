#ifndef PARSER_HPP
#define PARSER_HPP

#include <iostream>
#include <fstream>

#include "parkingSpot.hpp"

/**
 * Parses an XML file to extract information about parking spots and stores them in the provided vector.
 * Each parking spot is represented by its ID, occupancy status, and bounding box (a rotated rectangle).
 *
 * The XML file is expected to have a format with `<space>` tags that contain attributes like `id`, `occupied`,
 * `center` (with x and y coordinates), `size` (with width and height), and `angle` (rotation angle).
 *
 * @param filePath        The path to the XML file to be parsed.
 * @param parkingSpot     A reference to a vector of ParkingSpot objects where the parsed data will be stored.
 */
void parseXML(const std::string& filePath, std::vector<ParkingSpot> &ParkingSpot);

#endif // PARSER_HP
cmake_minimum_required(VERSION 3.10)
project(EasyParking)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Trova OpenCV
find_package(OpenCV REQUIRED)

# Imposta directory di inclusione generale
include_directories(include ${OpenCV_INCLUDE_DIRS})

# Raggruppa tutti i file sorgenti
file(GLOB_RECURSE SOURCES src/*.cpp)

# Definisci le librerie
add_library(parkingSpot
    include/parkingSpot.hpp
    src/parkingSpot.cpp
)
target_link_libraries(parkingSpot ${OpenCV_LIBS})

add_library(parkingSpotDetector
    include/parkingSpotDetector.hpp
    src/parkingSpotDetector.cpp
)
target_link_libraries(parkingSpotDetector ${OpenCV_LIBS})

add_library(parser
    include/parser.hpp
    src/parser.cpp
)
target_link_libraries(parser ${OpenCV_LIBS})

add_library(utils
    include/utils.hpp
    src/utils.cpp
)
target_link_libraries(utils ${OpenCV_LIBS})

add_library(segmentation
    include/segmentation.hpp
    src/segmentation.cpp
)
target_link_libraries(segmentation ${OpenCV_LIBS})

add_library(performanceMeasurement
    include/performanceMeasurement.hpp
    src/performanceMeasurement.cpp
)
target_link_libraries(performanceMeasurement ${OpenCV_LIBS})

add_library(classification
    include/classification.hpp
    src/classification.cpp
)
target_link_libraries(classification ${OpenCV_LIBS})

add_library(minimap
    include/minimap.hpp
    src/minimap.cpp
)
target_link_libraries(minimap ${OpenCV_LIBS})

add_library(parkingSpotUtils
    include/parkingSpotUtils.hpp
    src/parkingSpotUtils.cpp
)
target_link_libraries(parkingSpotUtils ${OpenCV_LIBS})

# Aggiungi l'eseguibile
add_executable(EasyParking ${SOURCES})

# Linka librerie all'eseguibile
target_link_libraries(EasyParking 
    parkingSpot
    parkingSpotDetector
    parser
    utils
    segmentation
    performanceMeasurement
    classification
    minimap
    parkingSpotUtils
    ${OpenCV_LIBS}
)
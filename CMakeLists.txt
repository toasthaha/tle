cmake_minimum_required(VERSION 2.8)
project(test)

find_package(OpenCV REQUIRED)

if(NOT WIN32)
ADD_DEFINITIONS("-std=c++0x -O3")
endif(NOT WIN32)

include_directories(src/tle) 
include_directories(src/tracker) 

add_subdirectory(src/tracker)

FILE(GLOB_RECURSE sourcefiles "src/tle/*.cpp")
add_executable( tle ${sourcefiles} )
target_link_libraries( tle ${OpenCV_LIBS})
target_link_libraries ( tle tracker)




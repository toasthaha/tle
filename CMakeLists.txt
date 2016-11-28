cmake_minimum_required(VERSION 2.8)
project(gamebler)

option(CPU_ONLY "Use CPU only for Caffe" OFF)
option(USE_CUDNN "Use cuDNN for Caffe" OFF)


find_package(OpenCV REQUIRED)

if(NOT WIN32)
ADD_DEFINITIONS("-std=c++0x -O3 -g")
endif(NOT WIN32)

if(USE_CUDNN)
  link_directories(~/cudnn-6.5-linux-R1)
endif()


include_directories(src/tle) 
include_directories(src/tracker) 
include_directories(/usr/include)
include_directories(~/work/dqn/caffe/include)
link_directories(~/work/dqn/caffe/build/lib)

add_subdirectory(src/tle)
add_subdirectory(src/tracker)

FILE(GLOB_RECURSE sourcefiles "src/dqn/*.cpp")
add_executable( dqn ${sourcefiles} )

target_link_libraries( dqn ${OpenCV_LIBS} )
target_link_libraries( dqn tracker )
target_link_libraries( dqn tle )
target_link_libraries( dqn caffe)
target_link_libraries( dqn glog)
target_link_libraries( dqn gflags)

if(USE_SDL)
  add_definitions(-D__USE_SDL)
  if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework Cocoa")
    target_link_libraries(dqn sdl)
    target_link_libraries(dqn sdl_gfx)
    target_link_libraries(dqn sdl_image)
    target_link_libraries(dqn sdlmain)
  else()
    target_link_libraries(dqn SDL)
    target_link_libraries(dqn SDL_gfx)
    target_link_libraries(dqn SDL_image)
    target_link_libraries(dqn SDLmain)
  endif()
endif()

if(NOT CPU_ONLY)
  include_directories(/usr/local/cuda/include)
endif()

if(CPU_ONLY)
  add_definitions(-DCPU_ONLY)
endif()

if(USE_CUDNN)
  target_link_libraries(dqn cudnn)
endif()

add_executable( runtle src/tle/runtle.cpp src/tle/tle_interface.cpp)
target_link_libraries( runtle tle)
target_link_libraries( runtle ${OpenCV_LIBS})
target_link_libraries( runtle tracker )


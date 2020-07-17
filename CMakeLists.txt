cmake_minimum_required(VERSION 3.0)
project(DroneProject) 

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

# These are just helper files
add_subdirectory(libfiles)
add_subdirectory(Camera)

# This directory contains the main file.
add_subdirectory(Program)

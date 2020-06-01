
# These two lines are also inside the CMakeLists.txt file in the Camera-folder 
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# These two lines are also inside the CMakeLists.txt file in the Camera-folder 
set (CMAKE_MODULE_PATH "/usr/local/lib/cmake/${CMAKE_MODULE_PATH}")
find_package(raspicam REQUIRED)
include_directories( ${raspicam_INCLUDE_DIRS} )

add_executable(mainprogram main.cpp )

target_link_libraries(mainprogram ${OpenCV_LIBS} )
target_link_libraries(mainprogram ${raspicam_CV_LIBS})

target_link_libraries(mainprogram CameraLib)
target_link_libraries(mainprogram Mylib)
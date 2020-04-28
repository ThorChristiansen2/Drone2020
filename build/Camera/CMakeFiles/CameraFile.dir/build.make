# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.7

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/pi/Drone2020

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/pi/Drone2020/build

# Include any dependencies generated for this target.
include Camera/CMakeFiles/CameraFile.dir/depend.make

# Include the progress variables for this target.
include Camera/CMakeFiles/CameraFile.dir/progress.make

# Include the compile flags for this target's objects.
include Camera/CMakeFiles/CameraFile.dir/flags.make

Camera/CMakeFiles/CameraFile.dir/main.cpp.o: Camera/CMakeFiles/CameraFile.dir/flags.make
Camera/CMakeFiles/CameraFile.dir/main.cpp.o: ../Camera/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pi/Drone2020/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object Camera/CMakeFiles/CameraFile.dir/main.cpp.o"
	cd /home/pi/Drone2020/build/Camera && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CameraFile.dir/main.cpp.o -c /home/pi/Drone2020/Camera/main.cpp

Camera/CMakeFiles/CameraFile.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CameraFile.dir/main.cpp.i"
	cd /home/pi/Drone2020/build/Camera && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pi/Drone2020/Camera/main.cpp > CMakeFiles/CameraFile.dir/main.cpp.i

Camera/CMakeFiles/CameraFile.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CameraFile.dir/main.cpp.s"
	cd /home/pi/Drone2020/build/Camera && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pi/Drone2020/Camera/main.cpp -o CMakeFiles/CameraFile.dir/main.cpp.s

Camera/CMakeFiles/CameraFile.dir/main.cpp.o.requires:

.PHONY : Camera/CMakeFiles/CameraFile.dir/main.cpp.o.requires

Camera/CMakeFiles/CameraFile.dir/main.cpp.o.provides: Camera/CMakeFiles/CameraFile.dir/main.cpp.o.requires
	$(MAKE) -f Camera/CMakeFiles/CameraFile.dir/build.make Camera/CMakeFiles/CameraFile.dir/main.cpp.o.provides.build
.PHONY : Camera/CMakeFiles/CameraFile.dir/main.cpp.o.provides

Camera/CMakeFiles/CameraFile.dir/main.cpp.o.provides.build: Camera/CMakeFiles/CameraFile.dir/main.cpp.o


# Object files for target CameraFile
CameraFile_OBJECTS = \
"CMakeFiles/CameraFile.dir/main.cpp.o"

# External object files for target CameraFile
CameraFile_EXTERNAL_OBJECTS =

Camera/CameraFile: Camera/CMakeFiles/CameraFile.dir/main.cpp.o
Camera/CameraFile: Camera/CMakeFiles/CameraFile.dir/build.make
Camera/CameraFile: /usr/local/lib/libopencv_dnn.so.4.3.0
Camera/CameraFile: /usr/local/lib/libopencv_gapi.so.4.3.0
Camera/CameraFile: /usr/local/lib/libopencv_highgui.so.4.3.0
Camera/CameraFile: /usr/local/lib/libopencv_ml.so.4.3.0
Camera/CameraFile: /usr/local/lib/libopencv_objdetect.so.4.3.0
Camera/CameraFile: /usr/local/lib/libopencv_photo.so.4.3.0
Camera/CameraFile: /usr/local/lib/libopencv_stitching.so.4.3.0
Camera/CameraFile: /usr/local/lib/libopencv_video.so.4.3.0
Camera/CameraFile: /usr/local/lib/libopencv_videoio.so.4.3.0
Camera/CameraFile: libfiles/libMylib.a
Camera/CameraFile: /usr/local/lib/libopencv_imgcodecs.so.4.3.0
Camera/CameraFile: /usr/local/lib/libopencv_calib3d.so.4.3.0
Camera/CameraFile: /usr/local/lib/libopencv_features2d.so.4.3.0
Camera/CameraFile: /usr/local/lib/libopencv_flann.so.4.3.0
Camera/CameraFile: /usr/local/lib/libopencv_imgproc.so.4.3.0
Camera/CameraFile: /usr/local/lib/libopencv_core.so.4.3.0
Camera/CameraFile: Camera/CMakeFiles/CameraFile.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/pi/Drone2020/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable CameraFile"
	cd /home/pi/Drone2020/build/Camera && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/CameraFile.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
Camera/CMakeFiles/CameraFile.dir/build: Camera/CameraFile

.PHONY : Camera/CMakeFiles/CameraFile.dir/build

Camera/CMakeFiles/CameraFile.dir/requires: Camera/CMakeFiles/CameraFile.dir/main.cpp.o.requires

.PHONY : Camera/CMakeFiles/CameraFile.dir/requires

Camera/CMakeFiles/CameraFile.dir/clean:
	cd /home/pi/Drone2020/build/Camera && $(CMAKE_COMMAND) -P CMakeFiles/CameraFile.dir/cmake_clean.cmake
.PHONY : Camera/CMakeFiles/CameraFile.dir/clean

Camera/CMakeFiles/CameraFile.dir/depend:
	cd /home/pi/Drone2020/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/pi/Drone2020 /home/pi/Drone2020/Camera /home/pi/Drone2020/build /home/pi/Drone2020/build/Camera /home/pi/Drone2020/build/Camera/CMakeFiles/CameraFile.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : Camera/CMakeFiles/CameraFile.dir/depend

# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

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
CMAKE_SOURCE_DIR = /home/tinhn/3D/ICCVW/Segmentator

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/tinhn/3D/ICCVW/Segmentator/build

# Include any dependencies generated for this target.
include CMakeFiles/segmentator.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/segmentator.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/segmentator.dir/flags.make

CMakeFiles/segmentator.dir/segmentator.cpp.o: CMakeFiles/segmentator.dir/flags.make
CMakeFiles/segmentator.dir/segmentator.cpp.o: ../segmentator.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tinhn/3D/ICCVW/Segmentator/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/segmentator.dir/segmentator.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/segmentator.dir/segmentator.cpp.o -c /home/tinhn/3D/ICCVW/Segmentator/segmentator.cpp

CMakeFiles/segmentator.dir/segmentator.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/segmentator.dir/segmentator.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tinhn/3D/ICCVW/Segmentator/segmentator.cpp > CMakeFiles/segmentator.dir/segmentator.cpp.i

CMakeFiles/segmentator.dir/segmentator.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/segmentator.dir/segmentator.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tinhn/3D/ICCVW/Segmentator/segmentator.cpp -o CMakeFiles/segmentator.dir/segmentator.cpp.s

CMakeFiles/segmentator.dir/tinyply.cpp.o: CMakeFiles/segmentator.dir/flags.make
CMakeFiles/segmentator.dir/tinyply.cpp.o: ../tinyply.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tinhn/3D/ICCVW/Segmentator/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/segmentator.dir/tinyply.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/segmentator.dir/tinyply.cpp.o -c /home/tinhn/3D/ICCVW/Segmentator/tinyply.cpp

CMakeFiles/segmentator.dir/tinyply.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/segmentator.dir/tinyply.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tinhn/3D/ICCVW/Segmentator/tinyply.cpp > CMakeFiles/segmentator.dir/tinyply.cpp.i

CMakeFiles/segmentator.dir/tinyply.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/segmentator.dir/tinyply.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tinhn/3D/ICCVW/Segmentator/tinyply.cpp -o CMakeFiles/segmentator.dir/tinyply.cpp.s

# Object files for target segmentator
segmentator_OBJECTS = \
"CMakeFiles/segmentator.dir/segmentator.cpp.o" \
"CMakeFiles/segmentator.dir/tinyply.cpp.o"

# External object files for target segmentator
segmentator_EXTERNAL_OBJECTS =

segmentator: CMakeFiles/segmentator.dir/segmentator.cpp.o
segmentator: CMakeFiles/segmentator.dir/tinyply.cpp.o
segmentator: CMakeFiles/segmentator.dir/build.make
segmentator: CMakeFiles/segmentator.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/tinhn/3D/ICCVW/Segmentator/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable segmentator"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/segmentator.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/segmentator.dir/build: segmentator

.PHONY : CMakeFiles/segmentator.dir/build

CMakeFiles/segmentator.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/segmentator.dir/cmake_clean.cmake
.PHONY : CMakeFiles/segmentator.dir/clean

CMakeFiles/segmentator.dir/depend:
	cd /home/tinhn/3D/ICCVW/Segmentator/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/tinhn/3D/ICCVW/Segmentator /home/tinhn/3D/ICCVW/Segmentator /home/tinhn/3D/ICCVW/Segmentator/build /home/tinhn/3D/ICCVW/Segmentator/build /home/tinhn/3D/ICCVW/Segmentator/build/CMakeFiles/segmentator.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/segmentator.dir/depend


# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/giovanni/Scrivania/EasyParking

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/giovanni/Scrivania/EasyParking/build

# Include any dependencies generated for this target.
include CMakeFiles/EasyParking.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/EasyParking.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/EasyParking.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/EasyParking.dir/flags.make

CMakeFiles/EasyParking.dir/src/main.cpp.o: CMakeFiles/EasyParking.dir/flags.make
CMakeFiles/EasyParking.dir/src/main.cpp.o: ../src/main.cpp
CMakeFiles/EasyParking.dir/src/main.cpp.o: CMakeFiles/EasyParking.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/giovanni/Scrivania/EasyParking/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/EasyParking.dir/src/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/EasyParking.dir/src/main.cpp.o -MF CMakeFiles/EasyParking.dir/src/main.cpp.o.d -o CMakeFiles/EasyParking.dir/src/main.cpp.o -c /home/giovanni/Scrivania/EasyParking/src/main.cpp

CMakeFiles/EasyParking.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/EasyParking.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/giovanni/Scrivania/EasyParking/src/main.cpp > CMakeFiles/EasyParking.dir/src/main.cpp.i

CMakeFiles/EasyParking.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/EasyParking.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/giovanni/Scrivania/EasyParking/src/main.cpp -o CMakeFiles/EasyParking.dir/src/main.cpp.s

# Object files for target EasyParking
EasyParking_OBJECTS = \
"CMakeFiles/EasyParking.dir/src/main.cpp.o"

# External object files for target EasyParking
EasyParking_EXTERNAL_OBJECTS =

EasyParking: CMakeFiles/EasyParking.dir/src/main.cpp.o
EasyParking: CMakeFiles/EasyParking.dir/build.make
EasyParking: CMakeFiles/EasyParking.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/giovanni/Scrivania/EasyParking/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable EasyParking"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/EasyParking.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/EasyParking.dir/build: EasyParking
.PHONY : CMakeFiles/EasyParking.dir/build

CMakeFiles/EasyParking.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/EasyParking.dir/cmake_clean.cmake
.PHONY : CMakeFiles/EasyParking.dir/clean

CMakeFiles/EasyParking.dir/depend:
	cd /home/giovanni/Scrivania/EasyParking/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/giovanni/Scrivania/EasyParking /home/giovanni/Scrivania/EasyParking /home/giovanni/Scrivania/EasyParking/build /home/giovanni/Scrivania/EasyParking/build /home/giovanni/Scrivania/EasyParking/build/CMakeFiles/EasyParking.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/EasyParking.dir/depend


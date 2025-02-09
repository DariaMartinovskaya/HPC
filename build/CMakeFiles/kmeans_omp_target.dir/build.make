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
CMAKE_SOURCE_DIR = /home/alunos/tei/2024/tei26610/hpc2/my-repo/hpc/repo

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/alunos/tei/2024/tei26610/hpc2/my-repo/hpc/build

# Include any dependencies generated for this target.
include CMakeFiles/kmeans_omp_target.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/kmeans_omp_target.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/kmeans_omp_target.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/kmeans_omp_target.dir/flags.make

CMakeFiles/kmeans_omp_target.dir/kmeans_omp_target.cpp.o: CMakeFiles/kmeans_omp_target.dir/flags.make
CMakeFiles/kmeans_omp_target.dir/kmeans_omp_target.cpp.o: /home/alunos/tei/2024/tei26610/hpc2/my-repo/hpc/repo/kmeans_omp_target.cpp
CMakeFiles/kmeans_omp_target.dir/kmeans_omp_target.cpp.o: CMakeFiles/kmeans_omp_target.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/alunos/tei/2024/tei26610/hpc2/my-repo/hpc/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/kmeans_omp_target.dir/kmeans_omp_target.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/kmeans_omp_target.dir/kmeans_omp_target.cpp.o -MF CMakeFiles/kmeans_omp_target.dir/kmeans_omp_target.cpp.o.d -o CMakeFiles/kmeans_omp_target.dir/kmeans_omp_target.cpp.o -c /home/alunos/tei/2024/tei26610/hpc2/my-repo/hpc/repo/kmeans_omp_target.cpp

CMakeFiles/kmeans_omp_target.dir/kmeans_omp_target.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kmeans_omp_target.dir/kmeans_omp_target.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/alunos/tei/2024/tei26610/hpc2/my-repo/hpc/repo/kmeans_omp_target.cpp > CMakeFiles/kmeans_omp_target.dir/kmeans_omp_target.cpp.i

CMakeFiles/kmeans_omp_target.dir/kmeans_omp_target.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kmeans_omp_target.dir/kmeans_omp_target.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/alunos/tei/2024/tei26610/hpc2/my-repo/hpc/repo/kmeans_omp_target.cpp -o CMakeFiles/kmeans_omp_target.dir/kmeans_omp_target.cpp.s

CMakeFiles/kmeans_omp_target.dir/kmeans_utils.cpp.o: CMakeFiles/kmeans_omp_target.dir/flags.make
CMakeFiles/kmeans_omp_target.dir/kmeans_utils.cpp.o: /home/alunos/tei/2024/tei26610/hpc2/my-repo/hpc/repo/kmeans_utils.cpp
CMakeFiles/kmeans_omp_target.dir/kmeans_utils.cpp.o: CMakeFiles/kmeans_omp_target.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/alunos/tei/2024/tei26610/hpc2/my-repo/hpc/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/kmeans_omp_target.dir/kmeans_utils.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/kmeans_omp_target.dir/kmeans_utils.cpp.o -MF CMakeFiles/kmeans_omp_target.dir/kmeans_utils.cpp.o.d -o CMakeFiles/kmeans_omp_target.dir/kmeans_utils.cpp.o -c /home/alunos/tei/2024/tei26610/hpc2/my-repo/hpc/repo/kmeans_utils.cpp

CMakeFiles/kmeans_omp_target.dir/kmeans_utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kmeans_omp_target.dir/kmeans_utils.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/alunos/tei/2024/tei26610/hpc2/my-repo/hpc/repo/kmeans_utils.cpp > CMakeFiles/kmeans_omp_target.dir/kmeans_utils.cpp.i

CMakeFiles/kmeans_omp_target.dir/kmeans_utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kmeans_omp_target.dir/kmeans_utils.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/alunos/tei/2024/tei26610/hpc2/my-repo/hpc/repo/kmeans_utils.cpp -o CMakeFiles/kmeans_omp_target.dir/kmeans_utils.cpp.s

# Object files for target kmeans_omp_target
kmeans_omp_target_OBJECTS = \
"CMakeFiles/kmeans_omp_target.dir/kmeans_omp_target.cpp.o" \
"CMakeFiles/kmeans_omp_target.dir/kmeans_utils.cpp.o"

# External object files for target kmeans_omp_target
kmeans_omp_target_EXTERNAL_OBJECTS =

kmeans_omp_target: CMakeFiles/kmeans_omp_target.dir/kmeans_omp_target.cpp.o
kmeans_omp_target: CMakeFiles/kmeans_omp_target.dir/kmeans_utils.cpp.o
kmeans_omp_target: CMakeFiles/kmeans_omp_target.dir/build.make
kmeans_omp_target: /usr/lib/gcc/x86_64-linux-gnu/12/libgomp.so
kmeans_omp_target: /usr/lib/x86_64-linux-gnu/libpthread.a
kmeans_omp_target: CMakeFiles/kmeans_omp_target.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/alunos/tei/2024/tei26610/hpc2/my-repo/hpc/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable kmeans_omp_target"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/kmeans_omp_target.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/kmeans_omp_target.dir/build: kmeans_omp_target
.PHONY : CMakeFiles/kmeans_omp_target.dir/build

CMakeFiles/kmeans_omp_target.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/kmeans_omp_target.dir/cmake_clean.cmake
.PHONY : CMakeFiles/kmeans_omp_target.dir/clean

CMakeFiles/kmeans_omp_target.dir/depend:
	cd /home/alunos/tei/2024/tei26610/hpc2/my-repo/hpc/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/alunos/tei/2024/tei26610/hpc2/my-repo/hpc/repo /home/alunos/tei/2024/tei26610/hpc2/my-repo/hpc/repo /home/alunos/tei/2024/tei26610/hpc2/my-repo/hpc/build /home/alunos/tei/2024/tei26610/hpc2/my-repo/hpc/build /home/alunos/tei/2024/tei26610/hpc2/my-repo/hpc/build/CMakeFiles/kmeans_omp_target.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/kmeans_omp_target.dir/depend


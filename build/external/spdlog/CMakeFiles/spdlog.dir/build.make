# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.31

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
CMAKE_COMMAND = /opt/homebrew/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/vasili/Desktop/Tests/Face_opencv

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/vasili/Desktop/Tests/Face_opencv/build

# Include any dependencies generated for this target.
include external/spdlog/CMakeFiles/spdlog.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include external/spdlog/CMakeFiles/spdlog.dir/compiler_depend.make

# Include the progress variables for this target.
include external/spdlog/CMakeFiles/spdlog.dir/progress.make

# Include the compile flags for this target's objects.
include external/spdlog/CMakeFiles/spdlog.dir/flags.make

external/spdlog/CMakeFiles/spdlog.dir/codegen:
.PHONY : external/spdlog/CMakeFiles/spdlog.dir/codegen

external/spdlog/CMakeFiles/spdlog.dir/src/spdlog.cpp.o: external/spdlog/CMakeFiles/spdlog.dir/flags.make
external/spdlog/CMakeFiles/spdlog.dir/src/spdlog.cpp.o: /Users/vasili/Desktop/Tests/Face_opencv/external/spdlog/src/spdlog.cpp
external/spdlog/CMakeFiles/spdlog.dir/src/spdlog.cpp.o: external/spdlog/CMakeFiles/spdlog.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/vasili/Desktop/Tests/Face_opencv/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object external/spdlog/CMakeFiles/spdlog.dir/src/spdlog.cpp.o"
	cd /Users/vasili/Desktop/Tests/Face_opencv/build/external/spdlog && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT external/spdlog/CMakeFiles/spdlog.dir/src/spdlog.cpp.o -MF CMakeFiles/spdlog.dir/src/spdlog.cpp.o.d -o CMakeFiles/spdlog.dir/src/spdlog.cpp.o -c /Users/vasili/Desktop/Tests/Face_opencv/external/spdlog/src/spdlog.cpp

external/spdlog/CMakeFiles/spdlog.dir/src/spdlog.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/spdlog.dir/src/spdlog.cpp.i"
	cd /Users/vasili/Desktop/Tests/Face_opencv/build/external/spdlog && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/vasili/Desktop/Tests/Face_opencv/external/spdlog/src/spdlog.cpp > CMakeFiles/spdlog.dir/src/spdlog.cpp.i

external/spdlog/CMakeFiles/spdlog.dir/src/spdlog.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/spdlog.dir/src/spdlog.cpp.s"
	cd /Users/vasili/Desktop/Tests/Face_opencv/build/external/spdlog && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/vasili/Desktop/Tests/Face_opencv/external/spdlog/src/spdlog.cpp -o CMakeFiles/spdlog.dir/src/spdlog.cpp.s

external/spdlog/CMakeFiles/spdlog.dir/src/stdout_sinks.cpp.o: external/spdlog/CMakeFiles/spdlog.dir/flags.make
external/spdlog/CMakeFiles/spdlog.dir/src/stdout_sinks.cpp.o: /Users/vasili/Desktop/Tests/Face_opencv/external/spdlog/src/stdout_sinks.cpp
external/spdlog/CMakeFiles/spdlog.dir/src/stdout_sinks.cpp.o: external/spdlog/CMakeFiles/spdlog.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/vasili/Desktop/Tests/Face_opencv/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object external/spdlog/CMakeFiles/spdlog.dir/src/stdout_sinks.cpp.o"
	cd /Users/vasili/Desktop/Tests/Face_opencv/build/external/spdlog && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT external/spdlog/CMakeFiles/spdlog.dir/src/stdout_sinks.cpp.o -MF CMakeFiles/spdlog.dir/src/stdout_sinks.cpp.o.d -o CMakeFiles/spdlog.dir/src/stdout_sinks.cpp.o -c /Users/vasili/Desktop/Tests/Face_opencv/external/spdlog/src/stdout_sinks.cpp

external/spdlog/CMakeFiles/spdlog.dir/src/stdout_sinks.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/spdlog.dir/src/stdout_sinks.cpp.i"
	cd /Users/vasili/Desktop/Tests/Face_opencv/build/external/spdlog && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/vasili/Desktop/Tests/Face_opencv/external/spdlog/src/stdout_sinks.cpp > CMakeFiles/spdlog.dir/src/stdout_sinks.cpp.i

external/spdlog/CMakeFiles/spdlog.dir/src/stdout_sinks.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/spdlog.dir/src/stdout_sinks.cpp.s"
	cd /Users/vasili/Desktop/Tests/Face_opencv/build/external/spdlog && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/vasili/Desktop/Tests/Face_opencv/external/spdlog/src/stdout_sinks.cpp -o CMakeFiles/spdlog.dir/src/stdout_sinks.cpp.s

external/spdlog/CMakeFiles/spdlog.dir/src/color_sinks.cpp.o: external/spdlog/CMakeFiles/spdlog.dir/flags.make
external/spdlog/CMakeFiles/spdlog.dir/src/color_sinks.cpp.o: /Users/vasili/Desktop/Tests/Face_opencv/external/spdlog/src/color_sinks.cpp
external/spdlog/CMakeFiles/spdlog.dir/src/color_sinks.cpp.o: external/spdlog/CMakeFiles/spdlog.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/vasili/Desktop/Tests/Face_opencv/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object external/spdlog/CMakeFiles/spdlog.dir/src/color_sinks.cpp.o"
	cd /Users/vasili/Desktop/Tests/Face_opencv/build/external/spdlog && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT external/spdlog/CMakeFiles/spdlog.dir/src/color_sinks.cpp.o -MF CMakeFiles/spdlog.dir/src/color_sinks.cpp.o.d -o CMakeFiles/spdlog.dir/src/color_sinks.cpp.o -c /Users/vasili/Desktop/Tests/Face_opencv/external/spdlog/src/color_sinks.cpp

external/spdlog/CMakeFiles/spdlog.dir/src/color_sinks.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/spdlog.dir/src/color_sinks.cpp.i"
	cd /Users/vasili/Desktop/Tests/Face_opencv/build/external/spdlog && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/vasili/Desktop/Tests/Face_opencv/external/spdlog/src/color_sinks.cpp > CMakeFiles/spdlog.dir/src/color_sinks.cpp.i

external/spdlog/CMakeFiles/spdlog.dir/src/color_sinks.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/spdlog.dir/src/color_sinks.cpp.s"
	cd /Users/vasili/Desktop/Tests/Face_opencv/build/external/spdlog && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/vasili/Desktop/Tests/Face_opencv/external/spdlog/src/color_sinks.cpp -o CMakeFiles/spdlog.dir/src/color_sinks.cpp.s

external/spdlog/CMakeFiles/spdlog.dir/src/file_sinks.cpp.o: external/spdlog/CMakeFiles/spdlog.dir/flags.make
external/spdlog/CMakeFiles/spdlog.dir/src/file_sinks.cpp.o: /Users/vasili/Desktop/Tests/Face_opencv/external/spdlog/src/file_sinks.cpp
external/spdlog/CMakeFiles/spdlog.dir/src/file_sinks.cpp.o: external/spdlog/CMakeFiles/spdlog.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/vasili/Desktop/Tests/Face_opencv/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object external/spdlog/CMakeFiles/spdlog.dir/src/file_sinks.cpp.o"
	cd /Users/vasili/Desktop/Tests/Face_opencv/build/external/spdlog && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT external/spdlog/CMakeFiles/spdlog.dir/src/file_sinks.cpp.o -MF CMakeFiles/spdlog.dir/src/file_sinks.cpp.o.d -o CMakeFiles/spdlog.dir/src/file_sinks.cpp.o -c /Users/vasili/Desktop/Tests/Face_opencv/external/spdlog/src/file_sinks.cpp

external/spdlog/CMakeFiles/spdlog.dir/src/file_sinks.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/spdlog.dir/src/file_sinks.cpp.i"
	cd /Users/vasili/Desktop/Tests/Face_opencv/build/external/spdlog && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/vasili/Desktop/Tests/Face_opencv/external/spdlog/src/file_sinks.cpp > CMakeFiles/spdlog.dir/src/file_sinks.cpp.i

external/spdlog/CMakeFiles/spdlog.dir/src/file_sinks.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/spdlog.dir/src/file_sinks.cpp.s"
	cd /Users/vasili/Desktop/Tests/Face_opencv/build/external/spdlog && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/vasili/Desktop/Tests/Face_opencv/external/spdlog/src/file_sinks.cpp -o CMakeFiles/spdlog.dir/src/file_sinks.cpp.s

external/spdlog/CMakeFiles/spdlog.dir/src/async.cpp.o: external/spdlog/CMakeFiles/spdlog.dir/flags.make
external/spdlog/CMakeFiles/spdlog.dir/src/async.cpp.o: /Users/vasili/Desktop/Tests/Face_opencv/external/spdlog/src/async.cpp
external/spdlog/CMakeFiles/spdlog.dir/src/async.cpp.o: external/spdlog/CMakeFiles/spdlog.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/vasili/Desktop/Tests/Face_opencv/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object external/spdlog/CMakeFiles/spdlog.dir/src/async.cpp.o"
	cd /Users/vasili/Desktop/Tests/Face_opencv/build/external/spdlog && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT external/spdlog/CMakeFiles/spdlog.dir/src/async.cpp.o -MF CMakeFiles/spdlog.dir/src/async.cpp.o.d -o CMakeFiles/spdlog.dir/src/async.cpp.o -c /Users/vasili/Desktop/Tests/Face_opencv/external/spdlog/src/async.cpp

external/spdlog/CMakeFiles/spdlog.dir/src/async.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/spdlog.dir/src/async.cpp.i"
	cd /Users/vasili/Desktop/Tests/Face_opencv/build/external/spdlog && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/vasili/Desktop/Tests/Face_opencv/external/spdlog/src/async.cpp > CMakeFiles/spdlog.dir/src/async.cpp.i

external/spdlog/CMakeFiles/spdlog.dir/src/async.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/spdlog.dir/src/async.cpp.s"
	cd /Users/vasili/Desktop/Tests/Face_opencv/build/external/spdlog && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/vasili/Desktop/Tests/Face_opencv/external/spdlog/src/async.cpp -o CMakeFiles/spdlog.dir/src/async.cpp.s

external/spdlog/CMakeFiles/spdlog.dir/src/cfg.cpp.o: external/spdlog/CMakeFiles/spdlog.dir/flags.make
external/spdlog/CMakeFiles/spdlog.dir/src/cfg.cpp.o: /Users/vasili/Desktop/Tests/Face_opencv/external/spdlog/src/cfg.cpp
external/spdlog/CMakeFiles/spdlog.dir/src/cfg.cpp.o: external/spdlog/CMakeFiles/spdlog.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/vasili/Desktop/Tests/Face_opencv/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object external/spdlog/CMakeFiles/spdlog.dir/src/cfg.cpp.o"
	cd /Users/vasili/Desktop/Tests/Face_opencv/build/external/spdlog && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT external/spdlog/CMakeFiles/spdlog.dir/src/cfg.cpp.o -MF CMakeFiles/spdlog.dir/src/cfg.cpp.o.d -o CMakeFiles/spdlog.dir/src/cfg.cpp.o -c /Users/vasili/Desktop/Tests/Face_opencv/external/spdlog/src/cfg.cpp

external/spdlog/CMakeFiles/spdlog.dir/src/cfg.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/spdlog.dir/src/cfg.cpp.i"
	cd /Users/vasili/Desktop/Tests/Face_opencv/build/external/spdlog && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/vasili/Desktop/Tests/Face_opencv/external/spdlog/src/cfg.cpp > CMakeFiles/spdlog.dir/src/cfg.cpp.i

external/spdlog/CMakeFiles/spdlog.dir/src/cfg.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/spdlog.dir/src/cfg.cpp.s"
	cd /Users/vasili/Desktop/Tests/Face_opencv/build/external/spdlog && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/vasili/Desktop/Tests/Face_opencv/external/spdlog/src/cfg.cpp -o CMakeFiles/spdlog.dir/src/cfg.cpp.s

external/spdlog/CMakeFiles/spdlog.dir/src/bundled_fmtlib_format.cpp.o: external/spdlog/CMakeFiles/spdlog.dir/flags.make
external/spdlog/CMakeFiles/spdlog.dir/src/bundled_fmtlib_format.cpp.o: /Users/vasili/Desktop/Tests/Face_opencv/external/spdlog/src/bundled_fmtlib_format.cpp
external/spdlog/CMakeFiles/spdlog.dir/src/bundled_fmtlib_format.cpp.o: external/spdlog/CMakeFiles/spdlog.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/vasili/Desktop/Tests/Face_opencv/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object external/spdlog/CMakeFiles/spdlog.dir/src/bundled_fmtlib_format.cpp.o"
	cd /Users/vasili/Desktop/Tests/Face_opencv/build/external/spdlog && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT external/spdlog/CMakeFiles/spdlog.dir/src/bundled_fmtlib_format.cpp.o -MF CMakeFiles/spdlog.dir/src/bundled_fmtlib_format.cpp.o.d -o CMakeFiles/spdlog.dir/src/bundled_fmtlib_format.cpp.o -c /Users/vasili/Desktop/Tests/Face_opencv/external/spdlog/src/bundled_fmtlib_format.cpp

external/spdlog/CMakeFiles/spdlog.dir/src/bundled_fmtlib_format.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/spdlog.dir/src/bundled_fmtlib_format.cpp.i"
	cd /Users/vasili/Desktop/Tests/Face_opencv/build/external/spdlog && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/vasili/Desktop/Tests/Face_opencv/external/spdlog/src/bundled_fmtlib_format.cpp > CMakeFiles/spdlog.dir/src/bundled_fmtlib_format.cpp.i

external/spdlog/CMakeFiles/spdlog.dir/src/bundled_fmtlib_format.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/spdlog.dir/src/bundled_fmtlib_format.cpp.s"
	cd /Users/vasili/Desktop/Tests/Face_opencv/build/external/spdlog && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/vasili/Desktop/Tests/Face_opencv/external/spdlog/src/bundled_fmtlib_format.cpp -o CMakeFiles/spdlog.dir/src/bundled_fmtlib_format.cpp.s

# Object files for target spdlog
spdlog_OBJECTS = \
"CMakeFiles/spdlog.dir/src/spdlog.cpp.o" \
"CMakeFiles/spdlog.dir/src/stdout_sinks.cpp.o" \
"CMakeFiles/spdlog.dir/src/color_sinks.cpp.o" \
"CMakeFiles/spdlog.dir/src/file_sinks.cpp.o" \
"CMakeFiles/spdlog.dir/src/async.cpp.o" \
"CMakeFiles/spdlog.dir/src/cfg.cpp.o" \
"CMakeFiles/spdlog.dir/src/bundled_fmtlib_format.cpp.o"

# External object files for target spdlog
spdlog_EXTERNAL_OBJECTS =

external/spdlog/libspdlog.a: external/spdlog/CMakeFiles/spdlog.dir/src/spdlog.cpp.o
external/spdlog/libspdlog.a: external/spdlog/CMakeFiles/spdlog.dir/src/stdout_sinks.cpp.o
external/spdlog/libspdlog.a: external/spdlog/CMakeFiles/spdlog.dir/src/color_sinks.cpp.o
external/spdlog/libspdlog.a: external/spdlog/CMakeFiles/spdlog.dir/src/file_sinks.cpp.o
external/spdlog/libspdlog.a: external/spdlog/CMakeFiles/spdlog.dir/src/async.cpp.o
external/spdlog/libspdlog.a: external/spdlog/CMakeFiles/spdlog.dir/src/cfg.cpp.o
external/spdlog/libspdlog.a: external/spdlog/CMakeFiles/spdlog.dir/src/bundled_fmtlib_format.cpp.o
external/spdlog/libspdlog.a: external/spdlog/CMakeFiles/spdlog.dir/build.make
external/spdlog/libspdlog.a: external/spdlog/CMakeFiles/spdlog.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/vasili/Desktop/Tests/Face_opencv/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Linking CXX static library libspdlog.a"
	cd /Users/vasili/Desktop/Tests/Face_opencv/build/external/spdlog && $(CMAKE_COMMAND) -P CMakeFiles/spdlog.dir/cmake_clean_target.cmake
	cd /Users/vasili/Desktop/Tests/Face_opencv/build/external/spdlog && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/spdlog.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
external/spdlog/CMakeFiles/spdlog.dir/build: external/spdlog/libspdlog.a
.PHONY : external/spdlog/CMakeFiles/spdlog.dir/build

external/spdlog/CMakeFiles/spdlog.dir/clean:
	cd /Users/vasili/Desktop/Tests/Face_opencv/build/external/spdlog && $(CMAKE_COMMAND) -P CMakeFiles/spdlog.dir/cmake_clean.cmake
.PHONY : external/spdlog/CMakeFiles/spdlog.dir/clean

external/spdlog/CMakeFiles/spdlog.dir/depend:
	cd /Users/vasili/Desktop/Tests/Face_opencv/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/vasili/Desktop/Tests/Face_opencv /Users/vasili/Desktop/Tests/Face_opencv/external/spdlog /Users/vasili/Desktop/Tests/Face_opencv/build /Users/vasili/Desktop/Tests/Face_opencv/build/external/spdlog /Users/vasili/Desktop/Tests/Face_opencv/build/external/spdlog/CMakeFiles/spdlog.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : external/spdlog/CMakeFiles/spdlog.dir/depend


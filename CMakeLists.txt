cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME FaceID)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)

add_executable(${PROJECT_NAME} 
                src/main.cpp 
                src/FaceRecognition/faceRecognition.cpp
                src/FaceRecognition/faceRecognition.h
                src/ImageProcessing/imageProcessing.cpp
                src/ImageProcessing/imageProcessing.h
                )

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/external/opencv/build/install/lib/cmake/opencv4) 
find_package(OpenCV REQUIRED)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

add_subdirectory(external/spdlog)
target_link_libraries(${PROJECT_NAME} spdlog)

# add_subdirectory(external/dlib)
# target_link_libraries(${PROJECT_NAME} dlib)

set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_directory
                    ${CMAKE_SOURCE_DIR}/resources $<TARGET_FILE_DIR:${PROJECT_NAME}>/resources)
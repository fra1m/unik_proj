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
                src/FaceEmbedding/faceEmbedding.h
                src/FaceEmbedding/faceEmbedding.cpp
                src/FaceMemory/faceMemory.h
                src/FaceMemory/faceMemory.cpp
                )

set(OpenCV_DIR /opt/homebrew/opt/opencv/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

add_executable(FaceIDCli
                src/cli.cpp
                src/FaceRecognition/faceRecognition.cpp
                src/FaceRecognition/faceRecognition.h
                src/ImageProcessing/imageProcessing.cpp
                src/ImageProcessing/imageProcessing.h
                src/FaceEmbedding/faceEmbedding.h
                src/FaceEmbedding/faceEmbedding.cpp
                )
target_link_libraries(FaceIDCli ${OpenCV_LIBS})

add_subdirectory(external/spdlog)
target_link_libraries(${PROJECT_NAME} spdlog)
target_link_libraries(FaceIDCli spdlog)

add_subdirectory(external/eigen)
target_link_libraries(${PROJECT_NAME} eigen)
target_link_libraries(FaceIDCli eigen)

add_subdirectory(external/dlib)
target_link_libraries(${PROJECT_NAME} dlib)
target_link_libraries(FaceIDCli dlib)

set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)
set_target_properties(FaceIDCli PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_directory
                    ${CMAKE_SOURCE_DIR}/resources $<TARGET_FILE_DIR:${PROJECT_NAME}>/resources)
add_custom_command(TARGET FaceIDCli POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_directory
                    ${CMAKE_SOURCE_DIR}/resources $<TARGET_FILE_DIR:FaceIDCli>/resources)

# build
# first step: cmake -S . -B build
# second step: cmake --build build

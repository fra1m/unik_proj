#include "imageProcessing.h"
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>

namespace fs = std::filesystem;

bool ImageProcessing::saveFaceImage(const cv::Mat &frame,
                                    const std::string &folder, int &counter) {
  fs::create_directories(folder);
  std::string path = folder + "/" + std::to_string(counter) + ".jpg";

  do {
    path = folder + "/" + std::to_string(counter) + ".jpg";
    counter++; // Увеличиваем счетчик для следующего файла
  } while (
      fs::exists(path)); // Проверяем, существует ли уже файл с таким именем

  if (cv::imwrite(path, frame)) {
    spdlog::info("Saved: {}", path);
    counter++;
    return true;
  } else {
    spdlog::error("Error saving image: {}", path);
    return false;
  }
}

void ImageProcessing::displayImage(const cv::Mat &frame) {
  cv::imshow("Detected Faces", frame);
  cv::waitKey(1);
}

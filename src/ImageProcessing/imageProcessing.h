#pragma once

#include <opencv2/opencv.hpp>
#include <string>

class ImageProcessing {
public:
  // Статический метод для сохранения изображений.
  static void saveFaceImage(const cv::Mat &frame, const std::string &folder,
                            int &counter);

private:
  ImageProcessing() = delete;
  ~ImageProcessing() = delete;
};

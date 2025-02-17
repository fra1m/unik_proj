#pragma once
#include <opencv2/opencv.hpp>
#include <string>

class ImageProcessing {
public:
  static bool saveFaceImage(const cv::Mat &frame, const std::string &folder,
                            int &counter);
  static void displayImage(const cv::Mat &frame);
};

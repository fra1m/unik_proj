#pragma once
#include "../FaceEmbedding/faceEmbedding.h" // Убедитесь, что путь правильный
#include <filesystem>
#include <opencv2/dnn.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <string>

class FaceRecognition {
public:
  FaceRecognition(const std::string &modelConfig,
                  const std::string &modelWeights,
                  FaceEmbedding &faceEmbedding);
  ~FaceRecognition();

  bool detectFace(cv::Mat &frame);
  void train(const std::string &positivePath, const std::string &negativePath);
  int predict(const cv::Mat &face);
  cv::Rect getLastFaceRegion();
  void loadSVM(const std::string &svmPath);

  bool isMyFace(const cv::Mat &face);

  std::vector<cv::Rect> detectFaces(cv::Mat &frame);

private:
  cv::dnn::Net net;
  cv::Rect faceRegion;
  const float CONFIDENCE_THRESHOLD = 0.5;
  cv::Ptr<cv::ml::SVM> svm;
  FaceEmbedding faceEmbedding; // Добавляем объект FaceEmbedding
};
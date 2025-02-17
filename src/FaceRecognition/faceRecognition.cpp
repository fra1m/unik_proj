#include "faceRecognition.h"
#include <filesystem>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>

cv::Rect lastFaceRegion; // Глобальная переменная для хранения координат

cv::Rect FaceRecognition::getLastFaceRegion() { return faceRegion; }

FaceRecognition::FaceRecognition(const std::string &modelConfig,
                                 const std::string &modelWeights) {
  net = cv::dnn::readNetFromCaffe(modelConfig, modelWeights);
  if (net.empty()) {
    std::cerr << "Ошибка загрузки модели!" << std::endl;
  }
}

bool FaceRecognition::detectFace(cv::Mat &frame) {
  if (frame.empty()) {
    spdlog::error("detectFace: Empty frame received!");
    return false;
  }

  cv::Mat blob =
      cv::dnn::blobFromImage(frame, 1.0, cv::Size(300, 300),
                             cv::Scalar(104.0, 177.0, 123.0), false, false);
  net.setInput(blob);
  cv::Mat detections = net.forward();

  cv::Mat detectionMat(detections.size[2], detections.size[3], CV_32F,
                       detections.ptr<float>());
  float maxConfidence = 0.0;
  cv::Rect bestFace;

  for (int i = 0; i < detectionMat.rows; i++) {
    float confidence = detectionMat.at<float>(i, 2);

    if (confidence > maxConfidence) {
      int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
      int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
      int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
      int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);
      bestFace = cv::Rect(x1, y1, x2 - x1, y2 - y1);
      maxConfidence = confidence;
    }
  }

  if (maxConfidence > CONFIDENCE_THRESHOLD) {
    this->faceRegion = bestFace;
    cv::rectangle(frame, bestFace, cv::Scalar(0, 255, 0), 2);
    return true;
  }

  return false;
}

void FaceRecognition::train(const std::string &positivePath,
                            const std::string &negativePath) {

  spdlog::info("Starting training using images from: {} and {}", positivePath,
               negativePath);

  std::vector<cv::Mat> images;
  std::vector<int> labels;

  // Загружаем положительные примеры (твои фото)
  for (const auto &entry : std::filesystem::directory_iterator(positivePath)) {
    cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
    if (img.empty())
      continue;
    cv::resize(img, img, cv::Size(300, 300));
    images.push_back(img);
    labels.push_back(1);
  }

  // Загружаем отрицательные примеры (другие люди)
  for (const auto &entry : std::filesystem::directory_iterator(negativePath)) {
    cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
    if (img.empty())
      continue;
    cv::resize(img, img, cv::Size(300, 300));
    images.push_back(img);
    labels.push_back(0); // Метка "Не я"
  }

  if (images.size() < 2) {
    spdlog::error("Not enough training images! Need at least two classes.");
    return;
  }

  // Преобразование изображений в признаки
  cv::Mat trainingData;
  for (const auto &img : images) {
    cv::Mat blob = cv::dnn::blobFromImage(
        img, 1.0, cv::Size(300, 300), cv::Scalar(104, 177, 123), false, false);
    net.setInput(blob);
    cv::Mat features = net.forward();
    features = features.reshape(1, 1);
    trainingData.push_back(features);

    spdlog::info("Размерность признаков при обучении: {} x {}", features.rows,
                 features.cols);
  }

  trainingData.convertTo(trainingData, CV_32F);
  cv::Mat labelsMat(labels.size(), 1, CV_32SC1, labels.data());

  // Создание и обучение SVM
  auto svm = cv::ml::SVM::create();
  svm->setType(cv::ml::SVM::C_SVC);
  svm->setKernel(cv::ml::SVM::LINEAR);
  svm->train(trainingData, cv::ml::ROW_SAMPLE, labelsMat);
  svm->save("face_svm.yml");

  spdlog::info("Training completed. Model saved to face_svm.yml");
}

int FaceRecognition::predict(const cv::Mat &face) {
  if (face.empty()) {
    spdlog::error("predict: Пустое изображение лица!");
    return -1;
  }
  if (svm.empty()) {
    spdlog::error("SVM не загружен!");
    return -1;
  }

  // Преобразование изображения в нужный формат
  cv::Mat face_resized;
  cv::resize(face, face_resized, cv::Size(300, 300));

  // Используем ту же нейросеть для извлечения признаков, что и в train()
  cv::Mat blob =
      cv::dnn::blobFromImage(face_resized, 1.0, cv::Size(300, 300),
                             cv::Scalar(104, 177, 123), false, false);
  net.setInput(blob);
  cv::Mat features = net.forward();
  features = features.reshape(1, 1);
  features.convertTo(features, CV_32F);

  if (features.cols != svm->getVarCount()) {
    spdlog::error("Размерность features ({}) не совпадает с var_count ({}).",
                  features.cols, svm->getVarCount());
    return -1;
  }

  // Предсказание SVM
  float response = svm->predict(features);
  return (response > 0) ? 1 : 0;
}

void FaceRecognition::loadSVM(const std::string &svmPath) {
  if (std::filesystem::exists(svmPath)) {
    svm = cv::ml::SVM::load(svmPath);
    spdlog::info("SVM модель загружена из {}", svmPath);
  } else {
    spdlog::warn("Файл модели SVM {} не найден.", svmPath);
  }
}

bool FaceRecognition::isMyFace(const cv::Mat &face) {
  int prediction = predict(face);
  spdlog::info("Face prediction: {}",
               prediction); // Логируем результат предсказания
  return prediction == 1;   // Предсказание 1 означает, что это ваше лицо
}
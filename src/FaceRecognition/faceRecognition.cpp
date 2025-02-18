#include "faceRecognition.h"
#include "../FaceEmbedding/faceEmbedding.h"

#include <dlib/matrix.h>
#include <filesystem>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>

cv::Rect lastFaceRegion; // Глобальная переменная для хранения координат
cv::Rect FaceRecognition::getLastFaceRegion() { return faceRegion; }

cv::Mat dlibMatrixToCvMat(const dlib::matrix<float, 0, 1> &dlibMat) {
  cv::Mat cvMat(1, dlibMat.size(), CV_32F); // Создаем однострочную матрицу
  for (long i = 0; i < dlibMat.size(); ++i) {
    cvMat.at<float>(0, i) = dlibMat(i); // Копируем данные
  }
  return cvMat;
}

FaceRecognition::FaceRecognition(const std::string &modelConfig,
                                 const std::string &modelWeights,
                                 FaceEmbedding &faceEmbedding)
    : faceEmbedding(faceEmbedding) { // Загрузка детектора лиц
  net = cv::dnn::readNetFromCaffe(modelConfig, modelWeights);

  // Пытаемся загрузить SVM при инициализации
  const std::string default_model_path = "face_svm.yml";
  if (std::filesystem::exists(default_model_path)) {
    loadSVM(default_model_path);
    spdlog::info("Автоматически загружена сохраненная модель");
  }
}

FaceRecognition::~FaceRecognition() {
  if (!svm->empty()) {
    svm->save("face_svm.yml");
    spdlog::info("Модель автоматически сохранена при выходе");
  }
}

bool FaceRecognition::detectFace(cv::Mat &frame) {
  if (frame.empty()) {
    spdlog::error("detectFace: Empty frame received!");
    return false;
  }

  cv::Mat color_frame;
  if (frame.channels() == 1) {
    cv::cvtColor(frame, color_frame, cv::COLOR_GRAY2BGR);
  } else {
    color_frame = frame.clone();
  }

  cv::Mat blob = cv::dnn::blobFromImage(color_frame, 1.0, cv::Size(300, 300),
                                        cv::Scalar(104.0, 177.0, 123.0),
                                        true, // swapRB = true
                                        false, CV_32F);

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
  spdlog::info(
      "Начало обучения модели с использованием изображений из: {} и {}",
      positivePath, negativePath);

  std::vector<cv::Mat> embeddings;
  std::vector<int> labels;

  // Загрузка положительных примеров
  for (const auto &entry : std::filesystem::directory_iterator(positivePath)) {
    cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
    if (img.channels() == 1) {
      cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
    }
    if (img.empty()) {
      spdlog::warn("Не удалось загрузить изображение: {}",
                   entry.path().string());
      continue;
    }

    // Получаем эмбеддинг через FaceEmbedding
    auto descriptors = faceEmbedding.getFaceDescriptor(img);
    if (descriptors.empty()) {
      spdlog::warn("Не удалось извлечь эмбеддинг для: {}",
                   entry.path().string());
      continue;
    }

    cv::Mat embedding = dlibMatrixToCvMat(descriptors[0]);
    embeddings.push_back(embedding);
    labels.push_back(1); // Метка "Я"
  }

  // Загрузка отрицательных примеров
  for (const auto &entry : std::filesystem::directory_iterator(negativePath)) {
    cv::Mat img = cv::imread(entry.path().string());
    if (img.empty()) {
      spdlog::warn("Не удалось загрузить изображение: {}",
                   entry.path().string());
      continue;
    }

    auto descriptors = faceEmbedding.getFaceDescriptor(img);
    if (descriptors.empty()) {
      spdlog::warn("Не удалось извлечь эмбеддинг для: {}",
                   entry.path().string());
      continue;
    }

    cv::Mat embedding = dlibMatrixToCvMat(descriptors[0]);
    embeddings.push_back(embedding);
    labels.push_back(0); // Метка "Не я"
  }

  if (embeddings.empty()) {
    spdlog::error("Нет данных для обучения!");
    return;
  }

  // Создание матрицы признаков
  cv::Mat trainingData;
  for (const auto &embedding : embeddings) {
    cv::Mat normEmbedding;
    // Нормализуем эмбеддинг
    cv::normalize(embedding, normEmbedding, 1.0, 0, cv::NORM_L2);
    trainingData.push_back(normEmbedding);
  }
  trainingData.convertTo(trainingData, CV_32F);

  cv::Mat labelsMat(labels.size(), 1, CV_32SC1, labels.data());

  // Создание и обучение SVM
  auto svm = cv::ml::SVM::create();
  svm->setType(cv::ml::SVM::C_SVC);
  svm->setKernel(cv::ml::SVM::LINEAR);

  try {
    svm->train(trainingData, cv::ml::ROW_SAMPLE, labelsMat);
    svm->save("face_svm.yml");
    spdlog::info("Обучение завершено. Модель сохранена.");
  } catch (const cv::Exception &e) {
    spdlog::error("Ошибка обучения SVM: {}", e.what());
  }
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

  // Получаем эмбеддинг через FaceEmbedding (как при обучении)
  auto descriptors = faceEmbedding.getFaceDescriptor(face);
  if (descriptors.empty()) {
    spdlog::error("Не удалось извлечь эмбеддинг для предсказания");
    return -1;
  }

  cv::Mat embedding = dlibMatrixToCvMat(descriptors[0]);
  // Применяем L2-нормализацию
  cv::normalize(embedding, embedding, 1.0, 0, cv::NORM_L2);

  if (embedding.cols != svm->getVarCount()) {
    spdlog::error("Размерность эмбеддинга ({}) не совпадает с SVM ({})",
                  embedding.cols, svm->getVarCount());
    return -1;
  }

  float response = svm->predict(embedding);
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

bool FaceRecognition::isMyFace(const cv::Mat &face_embedding) {
  if (svm.empty()) {
    spdlog::error("SVM модель не загружена!");
    return false;
  }

  // Проверка размерности эмбеддинга
  if (face_embedding.cols != svm->getVarCount()) {
    spdlog::error("Несоответствие размерности: эмбеддинг {} vs SVM {}",
                  face_embedding.cols, svm->getVarCount());
    return false;
  }

  // Прогноз с получением расстояния до разделяющей гиперплоскости
  cv::Mat results;
  float response =
      svm->predict(face_embedding, results, cv::ml::StatModel::RAW_OUTPUT);

  // Для бинарной классификации используем порог
  const float threshold = 0.0f; // Может потребовать настройки
  return results.at<float>(0) > threshold;
}
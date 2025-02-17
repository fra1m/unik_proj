#include "FaceRecognition/faceRecognition.h"
#include "ImageProcessing/imageProcessing.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>

int main() {
  // Загрузка модели
  std::string modelConfig = "../resources/dnn/deploy.prototxt";
  std::string modelWeights =
      "../resources/dnn/res10_300x300_ssd_iter_140000.caffemodel";

  spdlog::info("Loading model: {}", modelConfig);
  spdlog::info("Loading weights: {}", modelWeights);

  // Создание объекта для распознавания лиц
  FaceRecognition faceRecognition(modelConfig, modelWeights);
  faceRecognition.loadSVM("face_svm.yml"); // Загрузка обученной SVM модели

  // Инициализация камеры
  cv::VideoCapture cap(0);
  if (!cap.isOpened()) {
    std::cerr << "Error: Unable to access camera!" << std::endl;
    return -1;
  }

  int counter = 0;
  cv::Mat frame;

  std::cout << "Нажимайте 's' для сохранения изображения с лицом." << std::endl;
  std::cout << "Нажмите 't' для тренировки модели." << std::endl;
  std::cout << "Нажмите 'q' для выхода." << std::endl;

  int stableCount = 0;                // Счетчик стабильных кадров
  const int STABLE_THRESHOLD = 5;     // Порог стабилизации, например 5 кадров
  std::string lastStableText = "BAD"; // Последнее стабильное предсказание

  while (true) {
    cap >> frame;
    if (frame.empty()) {
      std::cerr << "Error: Empty frame captured!" << std::endl;
      break;
    }

    bool isFaceDetected = faceRecognition.detectFace(frame);
    std::string text =
        lastStableText; // По умолчанию текст = последнее стабильное значение

    // Добавим проверку на чужое лицо
    if (isFaceDetected) {
      cv::Rect faceRegion = faceRecognition.getLastFaceRegion();
      if (faceRegion.area() > 0) {
        cv::Mat faceROI = frame(faceRegion).clone();
        int prediction = faceRecognition.predict(faceROI);

        std::string currentText = "BAD"; // Начнем с BAD, если это чужое лицо

        if (faceRecognition.isMyFace(faceROI)) {
          currentText = "GOOD"; // Если это ваше лицо
        }

        if (currentText != lastStableText) {
          // Если лицо изменилось
          stableCount = 0;              // Сбросить счетчик
          lastStableText = currentText; // Обновить стабильный текст сразу
          spdlog::info("Face changed, setting text to {}", currentText);
        }

        // Обновить счетчик стабильности
        if (currentText == lastStableText) {
          stableCount++;
          if (stableCount >= STABLE_THRESHOLD) {
            spdlog::info(
                "stableCount reached threshold: {}, updating lastStableText",
                stableCount);
            lastStableText = currentText;
            stableCount = 0; // Сбросить счетчик после обновления
          }
        }
      }
    } else {
      // Лицо не обнаружено, обновить на NO FACES
      if (lastStableText != "NO FACES") {
        lastStableText = "NO FACES";
        spdlog::info("No face detected, updating text to: NO FACES");
      }
      stableCount = 0; // Сбросить счетчик, так как лицо исчезло
    }

    // Добавляем текст на изображение
    cv::putText(frame, lastStableText, cv::Point(50, 50),
                cv::FONT_HERSHEY_SIMPLEX, 1,
                (lastStableText == "GOOD")
                    ? cv::Scalar(0, 255, 0)
                    : (lastStableText == "NO FACES" ? cv::Scalar(255, 255, 0)
                                                    : cv::Scalar(0, 0, 255)),
                2);

    cv::imshow("Face Recognition", frame);

    // Обработка клавиш
    char key = cv::waitKey(1);

    if (key == 's') {
      if (isFaceDetected) {
        ImageProcessing::saveFaceImage(frame, "data/not_me", counter);
        spdlog::info("Face image saved.");
      } else {
        spdlog::warn("No face detected, skipping save.");
      }
    }

    if (key == 't') {
      faceRecognition.train("data/me", "data/not_me");
      spdlog::info("Training completed.");
    }

    if (key == 'q')
      break;
  }

  cap.release();
  cv::destroyAllWindows();
  return 0;
}
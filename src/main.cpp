#include "FaceEmbedding/faceEmbedding.h"
#include "FaceRecognition/faceRecognition.h"
#include "ImageProcessing/imageProcessing.h"
#include <chrono>
#include <dlib/matrix.h>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <thread>

int main() {
  // Инициализация faceEmbedding и faceRecognition
  const std::string model_path =
      "bin/resources/dlib/dlib_face_recognition_resnet_model_v1.dat";
  FaceEmbedding faceEmbedding(model_path);
  FaceRecognition faceRecognition(
      "../resources/dnn/deploy.prototxt",
      "../resources/dnn/res10_300x300_ssd_iter_140000.caffemodel",
      faceEmbedding);

  // Настройка видеозахвата
  cv::VideoCapture cap(0);
  cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
  cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
  if (!cap.isOpened()) {
    spdlog::error("Ошибка при открытии камеры!");
    return -1;
  }

  // Переменные для управления состоянием
  int processFrameInterval = 1;
  int frameCount = 0;
  std::string displayText = "NO FACE";
  bool showLandmarks = false; // Флаг для отображения ключевых точек

  while (true) {
    cv::Mat frame;
    cap >> frame;
    if (frame.empty()) {
      spdlog::error("Empty frame captured!");
      break;
    }

    cv::flip(frame, frame, 1);

    // Обработка клавиш
    char key = cv::waitKey(1);
    switch (key) {
    case 't': { // Обучение модели
      try {
        faceRecognition.train("data/me", "data/not_me");
        faceRecognition.loadSVM("face_svm.yml");
        spdlog::info("Модель успешно обучена и загружена.");
      } catch (const std::exception &e) {
        spdlog::error("Ошибка при обучении модели: {}", e.what());
      }
      break;
    }
    case 'q': { // Выход
      cap.release();
      cv::destroyAllWindows();
      return 0;
    }
    case 's': { // Сохранение изображения
      ImageProcessing::saveFaceImage(frame, "data/not_me", frameCount);
      break;
    }
    case 'm': { // Переключение отображения ключевых точек
      showLandmarks = !showLandmarks;
      spdlog::info("Отображение ключевых точек: {}",
                   showLandmarks ? "ВКЛ" : "ВЫКЛ");
      break;
    }
    default:
      break; // Игнорируем другие клавиши
    }

    // Основная обработка кадра
    if (frameCount % processFrameInterval == 0) {
      bool face_detected = faceRecognition.detectFace(frame);
      if (face_detected) {
        cv::Rect faceRegion = faceRecognition.getLastFaceRegion();
        if (faceRegion.area() > 0) {
          cv::Mat faceROI = frame(faceRegion).clone();
          int prediction = faceRecognition.predict(faceROI);
          displayText = (prediction == 1) ? "GOOD" : "BAD";
        } else {
          displayText = "NO FACE";
        }
      } else {
        displayText = "NO FACE";
      }
    }

    // Отрисовка ключевых точек, если включено
    if (showLandmarks) {
      auto facesData = faceEmbedding.getFaceData(frame);
      for (const auto &faceData : facesData) {
        // Все ключевые точки
        for (const auto &point : faceData.landmarks) {
          cv::circle(frame, cv::Point(point.x(), point.y()), 2,
                     cv::Scalar(0, 255, 255), -1);
        }

        // Глаза
        for (int i = 36; i <= 47; ++i) {
          cv::circle(
              frame,
              cv::Point(faceData.landmarks[i].x(), faceData.landmarks[i].y()),
              2, cv::Scalar(255, 0, 0), -1);
        }

        // Нос
        for (int i = 27; i <= 35; ++i) {
          cv::circle(
              frame,
              cv::Point(faceData.landmarks[i].x(), faceData.landmarks[i].y()),
              2, cv::Scalar(0, 0, 255), -1);
        }

        // Рот
        for (int i = 48; i <= 67; ++i) {
          cv::circle(
              frame,
              cv::Point(faceData.landmarks[i].x(), faceData.landmarks[i].y()),
              2, cv::Scalar(0, 255, 0), -1);
        }
      }
    }

    // Накладываем текст на изображение
    cv::putText(frame, displayText, cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX,
                1,
                (displayText == "GOOD")
                    ? cv::Scalar(0, 255, 0)
                    : (displayText == "NO FACE" ? cv::Scalar(255, 255, 0)
                                                : cv::Scalar(0, 0, 255)),
                2);

    // Отображение кадра
    cv::imshow("Webcam", frame);

    frameCount++;
  }

  cap.release();
  cv::destroyAllWindows();
  return 0;
}
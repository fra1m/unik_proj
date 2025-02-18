#include "FaceEmbedding/faceEmbedding.h"
#include "FaceRecognition/faceRecognition.h"
#include <dlib/matrix.h>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>

// Функция для преобразования dlib::matrix<float, 0, 1> в cv::Mat
inline cv::Mat dlibMatrixToCvMat(const dlib::matrix<float, 0, 1> &dlibMat) {
  cv::Mat cvMat(1, dlibMat.size(), CV_32F); // Создаем однострочную матрицу
  for (long i = 0; i < dlibMat.size(); ++i) {
    cvMat.at<float>(0, i) = dlibMat(i); // Копируем данные
  }
  return cvMat;
}

int main() {
  // Путь к модели dlib
  const std::string model_path =
      "bin/resources/dlib/dlib_face_recognition_resnet_model_v1.dat";
  FaceEmbedding faceEmbedding(model_path);

  // Пути к данным для обучения
  const std::string positivePath = "data/me";
  const std::string negativePath = "data/not_me";

  // Инициализация FaceRecognition
  FaceRecognition faceRecognition(
      "../resources/dnn/deploy.prototxt",
      "../resources/dnn/res10_300x300_ssd_iter_140000.caffemodel",
      faceEmbedding);

  // Открываем видеопоток с вебкамеры
  cv::VideoCapture cap(0);
  if (!cap.isOpened()) {
    spdlog::error("Ошибка при открытии камеры!");
    return -1;
  }

  while (true) {
    cv::Mat frame;
    cap >> frame;

    // Зеркальное отражение кадра для более естественного отображения
    cv::flip(frame, frame, 1);

    // Детекция лиц и отрисовка прямоугольников
    cv::Mat detection_frame = frame.clone();
    bool face_detected = faceRecognition.detectFace(detection_frame);

    // Только если лицо обнаружено, делаем классификацию
    if (face_detected) {
      // Получаем эмбеддинги лиц
      auto descriptors = faceEmbedding.getFaceDescriptor(frame);

      for (const auto &desc : descriptors) {
        cv::Mat embedding = dlibMatrixToCvMat(desc);
        if (faceRecognition.isMyFace(embedding)) {
          spdlog::info("Good - мое лицо");
          // Рисуем зеленый прямоугольник для "своего" лица
          cv::rectangle(frame, faceRecognition.getLastFaceRegion(),
                        cv::Scalar(0, 255, 0), 2);
        } else {
          spdlog::info("Bad - чужое лицо");
          // Рисуем красный прямоугольник для "чужого" лица
          cv::rectangle(frame, faceRecognition.getLastFaceRegion(),
                        cv::Scalar(0, 0, 255), 2);
        }
      }
    }

    // Отображаем изображение с вебкамеры
    cv::imshow("Webcam", frame);

    // Обработка нажатия клавиш
    char key = cv::waitKey(1);
    if (key == 't') { // Если нажата клавиша 't'
      try {
        // Обучение модели
        faceRecognition.train(negativePath, positivePath);
        // Загрузка обученной SVM модели
        faceRecognition.loadSVM("face_svm.yml");
        spdlog::info("Модель успешно обучена и загружена.");
      } catch (const std::exception &e) {
        spdlog::error("Ошибка при обучении модели: {}", e.what());
      }
    } else if (key == 'q') { // Выход при нажатии на клавишу 'q'
      break;
    }
  }

  return 0;
}
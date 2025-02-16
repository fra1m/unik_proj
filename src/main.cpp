#include "FaceRecognition/faceRecognition.h"
#include "ImageProcessing/imageProcessing.h"

#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>

using namespace spdlog;
using namespace cv;
using namespace std;
namespace fs = std::filesystem;

int main() {
  // Открываем камеру
  VideoCapture cap(0);
  if (!cap.isOpened()) {
    cout << "Error: Unable to access camera!" << endl;
    return -1;
  }

  info("[INFO::Main]");

  // Путь к XML-файлу каскадного классификатора
  string faceCascadePath =
      "../resources/haarcascades/haarcascade_frontalface_default.xml";

  // Путь к обученной модели
  string trainerPath =
      "data/me/trainer.yml"; // Путь к обученной модели распознавания лиц

  // Инициализируем объект для распознавания лиц
  FaceRecognition faceRecognition(faceCascadePath, trainerPath);

  int counter = 0;
  Mat frame;

  vector<Mat> images;
  vector<int> labels;

  // Главное меню
  cout << "Нажимайте 's' для сохранения изображения с лицом. Нажмите 't' для "
          "тренировки модели."
       << endl;
  cout << "Нажмите 'q' для выхода." << endl;

  // Проверяем, существует ли файл с моделью (trainer.yml)
  if (!fs::exists(trainerPath)) {
    cout << "Model not found! Training new model..." << endl;
    // Если модель не найдена, тренируем ее
    while (true) {
      cap >> frame; // Захватываем кадр с камеры
      if (frame.empty())
        break;

      // Проверяем нажатие клавиш
      char key = waitKey(1); // Получаем нажатую клавишу

      if (key == 's') {
        // Сохранение изображения с лицом
        cout << "Saving image..." << endl;
        string folder = "data/me"; // Поменяй на "data/others" для других людей
        ImageProcessing::saveFaceImage(frame, folder, counter);
        // Добавляем изображение в список для тренировки
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        images.push_back(gray);
        labels.push_back(0); // Метка для вашего лица
      }

      if (key == 't' && !images.empty()) {
        // Тренировка модели
        cout << "Training model..." << endl;
        faceRecognition.trainModel(images, labels); // Обучаем модель
        break; // После тренировки выходим из цикла
      }

      if (key == 'q') {
        break; // Выход по 'q'
      }

      // Показываем кадр с надписью
      imshow("Frame", frame);
    }
  } else {
    // Если файл найден, загружаем модель
    cout << "Model found! Loading..." << endl;
    faceRecognition.loadModel(trainerPath); // Передаем путь к модели
  }

  // Продолжаем распознавание лиц, если модель загружена
  while (true) {
    cap >> frame; // Захватываем кадр с камеры
    if (frame.empty())
      break;

    // Распознаем лицо на изображении
    bool isFaceDetected = faceRecognition.detectAndRecognize(frame);

    // Отображаем результаты
    if (isFaceDetected) {
      putText(frame, "GOOD", Point(20, 50), FONT_HERSHEY_SIMPLEX, 1,
              Scalar(0, 255, 0), 2);
    } else {
      putText(frame, "BAD", Point(20, 50), FONT_HERSHEY_SIMPLEX, 1,
              Scalar(0, 0, 255), 2);
    }

    // Показываем кадр с результатами
    imshow("Frame", frame);

    // Проверяем нажатие клавиш
    char key = waitKey(1); // Получаем нажатую клавишу
    if (key == 'q') {
      break; // Выход по 'q'
    }
  }

  cap.release();       // Освобождаем ресурсы камеры
  destroyAllWindows(); // Закрываем все окна OpenCV

  return 0;
}

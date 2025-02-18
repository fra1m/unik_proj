#include "faceEmbedding.h"
#include <filesystem>
#include <iostream>
#include <spdlog/spdlog.h>

FaceEmbedding::FaceEmbedding(const std::string &model_path) {
  // Выводим текущую рабочую директорию
  spdlog::info("Текущая рабочая директория: {}",
               std::filesystem::current_path().string());

  // Строим полный путь к файлу модели
  std::string full_model_path =
      (std::filesystem::current_path() / model_path).string();
  spdlog::info("Полный путь к модели распознавания лиц: {}", full_model_path);

  // Проверяем, существует ли файл модели
  if (!std::filesystem::exists(full_model_path)) {
    spdlog::error("Файл модели распознавания лиц не найден: {}",
                  full_model_path);
    throw std::runtime_error("Файл модели не найден");
  }

  // Загрузка модели распознавания лиц
  try {
    dlib::deserialize(full_model_path) >> net;
    spdlog::info("Модель распознавания лиц успешно загружена.");
  } catch (const dlib::serialization_error &e) {
    spdlog::error("Ошибка загрузки модели распознавания лиц: {}", e.what());
    throw;
  }

  // Инициализация детектора лиц
  detector = dlib::get_frontal_face_detector();
  spdlog::info("Детектор лиц инициализирован.");

  // Строим полный путь к модели для ключевых точек
  std::string shape_predictor_path =
      (std::filesystem::current_path() /
       "bin/resources/dlib/shape_predictor_68_face_landmarks.dat")
          .string();
  spdlog::info("Полный путь к модели для ключевых точек: {}",
               shape_predictor_path);

  // Проверяем, существует ли файл модели для ключевых точек
  if (!std::filesystem::exists(shape_predictor_path)) {
    spdlog::error("Файл модели для ключевых точек не найден: {}",
                  shape_predictor_path);
    throw std::runtime_error("Файл модели не найден");
  }

  // Загрузка модели для ключевых точек (shape_predictor)
  try {
    dlib::deserialize(shape_predictor_path) >> sp;
    spdlog::info("Модель для ключевых точек успешно загружена.");
  } catch (const dlib::serialization_error &e) {
    spdlog::error("Ошибка загрузки модели для ключевых точек: {}", e.what());
    throw;
  }
}

std::vector<dlib::matrix<float, 0, 1>>
FaceEmbedding::getFaceDescriptor(const cv::Mat &frame) {
  std::vector<dlib::matrix<float, 0, 1>> face_descriptors;
  try {
    // Конвертируем OpenCV Mat в dlib::cv_image
    dlib::cv_image<dlib::bgr_pixel> dlibImg(frame);

    // Детекция лиц с помощью dlib
    std::vector<dlib::rectangle> faces = detector(dlibImg);

    // Извлечение признаков для каждого лица
    for (const auto &face : faces) {
      // Получаем ключевые точки лица
      dlib::full_object_detection shape = sp(dlibImg, face);

      // Срезаем лицо из изображения
      dlib::matrix<dlib::rgb_pixel> face_chip;
      dlib::extract_image_chip(
          dlibImg, dlib::get_face_chip_details(shape, 150, 0.25), face_chip);

      // Получаем эмбеддинг лица
      dlib::matrix<float, 0, 1> face_descriptor = net(face_chip);
      face_descriptors.push_back(face_descriptor);
      for (auto &desc : face_descriptors) {
        if (desc.size() != 128) { // Для ResNet модели
          spdlog::warn("Некорректный размер эмбеддинга: {}", desc.size());
          face_descriptors.clear();
          break;
        }
      }
    }
  } catch (const std::exception &e) {
    spdlog::error("Ошибка в getFaceDescriptor: {}", e.what());
  }

  return face_descriptors;
}
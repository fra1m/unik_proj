#include "imageProcessing.h"
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

void ImageProcessing::saveFaceImage(const cv::Mat &frame,
                                    const std::string &folder, int &counter) {
  // Создаем папку, если ее нет.
  fs::create_directories(folder);

  // Генерируем путь к файлу, добавляем индекс для предотвращения перезаписи
  std::string path;
  do {
    path = folder + "/" + std::to_string(counter) + ".jpg";
    counter++; // Увеличиваем счетчик для следующего файла
  } while (
      fs::exists(path)); // Проверяем, существует ли уже файл с таким именем

  // Сохраняем изображение в файл.
  if (cv::imwrite(path, frame)) {
    std::cout << "Saved: " << path << std::endl;
  } else {
    std::cerr << "Error saving image: " << path << std::endl;
  }
}

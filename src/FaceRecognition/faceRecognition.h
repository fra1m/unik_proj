#pragma once

#include <opencv2/face.hpp>
#include <opencv2/opencv.hpp>
#include <string>

using namespace cv;
using namespace std;

class FaceRecognition {
public:
  // Конструктор и деструктор
  FaceRecognition(const string &cascadePath, const string &trainerPath);
  ~FaceRecognition();

  // Загрузка модели
  bool loadModel(const string &trainerPath);

  // Обучение модели (если нужно)
  void trainModel(const vector<Mat> &images, const vector<int> &labels);

  // Детектирование и распознавание лица
  bool detectAndRecognize(Mat &frame);

private:
  CascadeClassifier faceCascade;
  Ptr<face::FaceRecognizer> faceRecognizer;
};

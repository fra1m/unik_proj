#include "faceRecognition.h"
#include <iostream>

FaceRecognition::FaceRecognition(const string &cascadePath,
                                 const string &trainerPath)
    : faceRecognizer(face::LBPHFaceRecognizer::create()) {
  // Загрузка каскадного классификатора
  if (!faceCascade.load(cascadePath)) {
    std::cerr << "Error: Unable to load cascade classifier!" << std::endl;
  }

  // Загружаем модель распознавания лица
  if (!loadModel(trainerPath)) {
    std::cerr << "Error: Unable to load face recognition model!" << std::endl;
  }
}

FaceRecognition::~FaceRecognition() { faceRecognizer.release(); }

bool FaceRecognition::loadModel(const string &trainerPath) {
  try {
    faceRecognizer->read(trainerPath);
    std::cout << "Model loaded successfully!" << std::endl;
    return true;
  } catch (const cv::Exception &e) {
    std::cerr << "Error loading model: " << e.what() << std::endl;
    return false;
  }
}

void FaceRecognition::trainModel(const vector<Mat> &images,
                                 const vector<int> &labels) {
  faceRecognizer->train(images, labels);
  faceRecognizer->save("data/me/trainer.yml");
}

bool FaceRecognition::detectAndRecognize(Mat &frame) {
  vector<Rect> faces;
  Mat gray;
  cvtColor(frame, gray, COLOR_BGR2GRAY);
  faceCascade.detectMultiScale(gray, faces);

  for (size_t i = 0; i < faces.size(); ++i) {
    Mat faceROI = gray(faces[i]);

    // Признаки лица
    int label = -1;
    double confidence = 0.0;

    // Распознаем лицо
    faceRecognizer->predict(faceROI, label, confidence);

    if (confidence < 100) {
      putText(frame, "GOOD", faces[i].tl(), FONT_HERSHEY_SIMPLEX, 1,
              Scalar(0, 255, 0), 2);
    } else {
      putText(frame, "BAD", faces[i].tl(), FONT_HERSHEY_SIMPLEX, 1,
              Scalar(0, 0, 255), 2);
    }

    rectangle(frame, faces[i], Scalar(0, 255, 0), 2);
  }

  return !faces.empty();
}

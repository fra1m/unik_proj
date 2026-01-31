#include "faceEmbedding.h"

#include <array>
#include <filesystem>
#include <iostream>
#include <spdlog/spdlog.h>

namespace fs = std::filesystem;

namespace {
const std::array<cv::Point2f, 5> kArcFaceTemplate = {
    cv::Point2f(38.2946f, 51.6963f), cv::Point2f(73.5318f, 51.5014f),
    cv::Point2f(56.0252f, 71.7366f), cv::Point2f(41.5493f, 92.3655f),
    cv::Point2f(70.7299f, 92.2041f)};

cv::Point2f meanPoints(const dlib::full_object_detection &shape, int start,
                       int end) {
  float sumX = 0.0f;
  float sumY = 0.0f;
  const int count = end - start + 1;
  for (int i = start; i <= end; ++i) {
    sumX += static_cast<float>(shape.part(i).x());
    sumY += static_cast<float>(shape.part(i).y());
  }
  return cv::Point2f(sumX / count, sumY / count);
}

bool extractArcFacePoints(const dlib::full_object_detection &shape,
                          std::array<cv::Point2f, 5> &out) {
  if (shape.num_parts() < 68) {
    return false;
  }

  out[0] = meanPoints(shape, 36, 41); // left eye
  out[1] = meanPoints(shape, 42, 47); // right eye
  out[2] = cv::Point2f(static_cast<float>(shape.part(30).x()),
                       static_cast<float>(shape.part(30).y())); // nose
  out[3] = cv::Point2f(static_cast<float>(shape.part(48).x()),
                       static_cast<float>(shape.part(48).y())); // left mouth
  out[4] = cv::Point2f(static_cast<float>(shape.part(54).x()),
                       static_cast<float>(shape.part(54).y())); // right mouth

  return true;
}

bool alignArcFace(const cv::Mat &frame,
                  const dlib::full_object_detection &shape,
                  cv::Mat &aligned) {
  std::array<cv::Point2f, 5> src;
  if (!extractArcFacePoints(shape, src)) {
    return false;
  }

  std::vector<cv::Point2f> srcVec(src.begin(), src.end());
  std::vector<cv::Point2f> dstVec(kArcFaceTemplate.begin(),
                                  kArcFaceTemplate.end());

  cv::Mat transform = cv::estimateAffinePartial2D(srcVec, dstVec);
  if (transform.empty()) {
    return false;
  }

  cv::warpAffine(frame, aligned, transform, cv::Size(112, 112),
                 cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                 cv::Scalar(0, 0, 0));
  return true;
}

dlib::matrix<float, 0, 1> cvMatToDlib(const cv::Mat &mat) {
  dlib::matrix<float, 0, 1> result;
  if (mat.empty()) {
    return result;
  }

  cv::Mat mat32;
  mat.convertTo(mat32, CV_32F);
  const int total = mat32.rows * mat32.cols;
  result.set_size(total);

  int index = 0;
  for (int r = 0; r < mat32.rows; ++r) {
    for (int c = 0; c < mat32.cols; ++c) {
      result(index++) = mat32.at<float>(r, c);
    }
  }
  return result;
}

bool normalizeEmbedding(dlib::matrix<float, 0, 1> &embedding) {
  const double norm = dlib::length(embedding);
  if (norm == 0.0) {
    return false;
  }
  embedding /= norm;
  return true;
}
} // namespace

FaceEmbedding::FaceEmbedding(const std::string &model_path) {
  spdlog::info("Working directory: {}",
               fs::current_path().string());

  const fs::path full_model_path = fs::current_path() / model_path;
  spdlog::info("Embedding model path: {}", full_model_path.string());

  if (!fs::exists(full_model_path)) {
    spdlog::error("Embedding model not found: {}", full_model_path.string());
    throw std::runtime_error("Embedding model not found");
  }

  const std::string ext = full_model_path.extension().string();
  if (ext == ".onnx") {
    embedder_type_ = EmbedderType::ArcFaceOnnx;
    arcface_net_ = cv::dnn::readNetFromONNX(full_model_path.string());
    if (arcface_net_.empty()) {
      throw std::runtime_error("Failed to load ArcFace ONNX model");
    }
    spdlog::info("ArcFace ONNX model loaded.");
    embedding_dim_ = 512;
  } else {
    embedder_type_ = EmbedderType::DlibResNet;
    try {
      dlib::deserialize(full_model_path.string()) >> net;
      spdlog::info("Dlib ResNet model loaded.");
      embedding_dim_ = 128;
    } catch (const dlib::serialization_error &e) {
      spdlog::error("Failed to load dlib model: {}", e.what());
      throw;
    }
  }

  detector = dlib::get_frontal_face_detector();

  const fs::path shape_predictor_path =
      fs::current_path() /
      "resources/dlib/shape_predictor_68_face_landmarks.dat";
  spdlog::info("Shape predictor path: {}", shape_predictor_path.string());

  if (!fs::exists(shape_predictor_path)) {
    spdlog::error("Shape predictor not found: {}",
                  shape_predictor_path.string());
    throw std::runtime_error("Shape predictor not found");
  }

  try {
    dlib::deserialize(shape_predictor_path.string()) >> sp;
    spdlog::info("Shape predictor loaded.");
  } catch (const dlib::serialization_error &e) {
    spdlog::error("Failed to load shape predictor: {}", e.what());
    throw;
  }
}

std::vector<dlib::matrix<float, 0, 1>>
FaceEmbedding::getFaceDescriptor(const cv::Mat &frame) {
  std::vector<dlib::matrix<float, 0, 1>> face_descriptors;
  if (frame.empty()) {
    return face_descriptors;
  }

  try {
    dlib::cv_image<dlib::bgr_pixel> dlibImg(frame);
    std::vector<dlib::rectangle> faces = detector(dlibImg);

    for (const auto &face : faces) {
      const cv::Rect rect(static_cast<int>(face.left()),
                          static_cast<int>(face.top()),
                          static_cast<int>(face.width()),
                          static_cast<int>(face.height()));
      dlib::matrix<float, 0, 1> descriptor;
      if (getFaceDescriptor(frame, rect, descriptor)) {
        face_descriptors.push_back(descriptor);
      }
    }
  } catch (const std::exception &e) {
    spdlog::error("getFaceDescriptor failed: {}", e.what());
  }

  return face_descriptors;
}

bool FaceEmbedding::getFaceDescriptor(const cv::Mat &frame,
                                      const cv::Rect &faceRect,
                                      dlib::matrix<float, 0, 1> &descriptor) {
  if (frame.empty()) {
    return false;
  }

  const cv::Rect bounded =
      faceRect & cv::Rect(0, 0, frame.cols, frame.rows);
  if (bounded.width <= 0 || bounded.height <= 0) {
    return false;
  }

  try {
    dlib::cv_image<dlib::bgr_pixel> dlibImg(frame);
    dlib::rectangle rect(static_cast<long>(bounded.x),
                         static_cast<long>(bounded.y),
                         static_cast<long>(bounded.x + bounded.width - 1),
                         static_cast<long>(bounded.y + bounded.height - 1));

    dlib::full_object_detection shape = sp(dlibImg, rect);

    if (embedder_type_ == EmbedderType::ArcFaceOnnx) {
      cv::Mat aligned;
      if (!alignArcFace(frame, shape, aligned)) {
        return false;
      }

      cv::Mat blob = cv::dnn::blobFromImage(
          aligned, 1.0 / 128.0, cv::Size(112, 112),
          cv::Scalar(127.5, 127.5, 127.5), true, false);
      arcface_net_.setInput(blob);
      cv::Mat output = arcface_net_.forward();

      output = output.reshape(1, 1);
      descriptor = cvMatToDlib(output);
      if (descriptor.size() == 0) {
        return false;
      }
      if (!normalizeEmbedding(descriptor)) {
        return false;
      }
    } else {
      dlib::matrix<dlib::rgb_pixel> face_chip;
      dlib::extract_image_chip(
          dlibImg, dlib::get_face_chip_details(shape, 150, 0.25), face_chip);
      descriptor = net(face_chip);
    }

    if (descriptor.size() == 0) {
      return false;
    }
    if (embedding_dim_ == 0) {
      embedding_dim_ = static_cast<int>(descriptor.size());
    } else if (descriptor.size() != embedding_dim_) {
      spdlog::warn("Embedding dim changed from {} to {}", embedding_dim_,
                   descriptor.size());
      embedding_dim_ = static_cast<int>(descriptor.size());
    }
  } catch (const std::exception &e) {
    spdlog::error("getFaceDescriptor(rect) failed: {}", e.what());
    return false;
  }

  return true;
}

int FaceEmbedding::embeddingDim() const { return embedding_dim_; }

std::vector<FaceEmbedding::FaceData>
FaceEmbedding::getFaceData(const cv::Mat &frame) {
  std::vector<FaceData> result;
  if (frame.empty()) {
    return result;
  }

  dlib::cv_image<dlib::bgr_pixel> dlibImg(frame);
  auto faces = detector(dlibImg);

  for (const auto &face : faces) {
    const cv::Rect rect(static_cast<int>(face.left()),
                        static_cast<int>(face.top()),
                        static_cast<int>(face.width()),
                        static_cast<int>(face.height()));
    FaceData data;
    if (getFaceData(frame, rect, data)) {
      result.push_back(data);
    }
  }
  return result;
}

bool FaceEmbedding::getFaceData(const cv::Mat &frame,
                                const cv::Rect &faceRect, FaceData &data) {
  if (frame.empty()) {
    return false;
  }

  const cv::Rect bounded =
      faceRect & cv::Rect(0, 0, frame.cols, frame.rows);
  if (bounded.width <= 0 || bounded.height <= 0) {
    return false;
  }

  try {
    dlib::cv_image<dlib::bgr_pixel> dlibImg(frame);
    dlib::rectangle rect(static_cast<long>(bounded.x),
                         static_cast<long>(bounded.y),
                         static_cast<long>(bounded.x + bounded.width - 1),
                         static_cast<long>(bounded.y + bounded.height - 1));

    dlib::full_object_detection shape = sp(dlibImg, rect);
    data.landmarks.clear();
    for (unsigned i = 0; i < shape.num_parts(); ++i) {
      data.landmarks.push_back(shape.part(i));
    }

    if (!getFaceDescriptor(frame, faceRect, data.embedding)) {
      return false;
    }
  } catch (const std::exception &e) {
    spdlog::error("getFaceData(rect) failed: {}", e.what());
    return false;
  }

  return true;
}

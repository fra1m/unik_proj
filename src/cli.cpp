#include "FaceEmbedding/faceEmbedding.h"
#include "FaceRecognition/faceRecognition.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>

namespace {

constexpr float kRingStartAngle = -static_cast<float>(CV_PI) / 2.0f;

cv::Rect pickLargestFace(const std::vector<cv::Rect> &faces) {
  if (faces.empty()) {
    return {};
  }
  return *std::max_element(
      faces.begin(), faces.end(),
      [](const cv::Rect &a, const cv::Rect &b) { return a.area() < b.area(); });
}

std::string makeEmbeddingJson(const dlib::matrix<float, 0, 1> &embedding) {
  std::string json = "{\n  \"embedding\": [";
  for (long i = 0; i < embedding.size(); ++i) {
    if (i > 0) {
      json += ", ";
    }
    json += std::to_string(embedding(i));
  }
  json += "]";
  json += "\n}\n";
  return json;
}

struct HeadPose {
  float yaw = 0.0f;
  float pitch = 0.0f;
  float roll = 0.0f;
  bool valid = false;
};

struct PoseEstimate {
  int slot = -1;
  float magnitude = 0.0f;
  float yawNorm = 0.0f;
  float pitchNorm = 0.0f;
  bool valid = false;
};

struct PoseSummary {
  bool valid = false;
  float yawNorm = 0.0f;
  float pitchNorm = 0.0f;
  float magnitude = 0.0f;
  int slot = -1;
};

float clampf(float value, float minValue, float maxValue) {
  return std::max(minValue, std::min(value, maxValue));
}

cv::Point2f toPoint(const dlib::point &pt) {
  return cv::Point2f(static_cast<float>(pt.x()), static_cast<float>(pt.y()));
}

cv::Point2f meanPoints(const std::vector<dlib::point> &pts, int start,
                       int end) {
  float sumX = 0.0f;
  float sumY = 0.0f;
  const int count = end - start + 1;
  for (int i = start; i <= end; ++i) {
    sumX += static_cast<float>(pts[i].x());
    sumY += static_cast<float>(pts[i].y());
  }
  return cv::Point2f(sumX / count, sumY / count);
}

HeadPose estimateHeadPose(const FaceEmbedding::FaceData &data,
                          const cv::Size &frameSize) {
  HeadPose pose;
  if (data.landmarks.size() < 68) {
    return pose;
  }

  const std::vector<cv::Point3f> modelPoints = {
      {0.0f, 0.0f, 0.0f},
      {0.0f, -330.0f, -65.0f},
      {-225.0f, 170.0f, -135.0f},
      {225.0f, 170.0f, -135.0f},
      {-150.0f, -150.0f, -125.0f},
      {150.0f, -150.0f, -125.0f}};

  const std::vector<cv::Point2f> imagePoints = {
      toPoint(data.landmarks[30]),
      toPoint(data.landmarks[8]),
      toPoint(data.landmarks[36]),
      toPoint(data.landmarks[45]),
      toPoint(data.landmarks[48]),
      toPoint(data.landmarks[54])};

  const double focalLength = static_cast<double>(frameSize.width);
  const cv::Point2d center(frameSize.width / 2.0, frameSize.height / 2.0);
  const cv::Mat cameraMatrix =
      (cv::Mat_<double>(3, 3) << focalLength, 0.0, center.x, 0.0, focalLength,
       center.y, 0.0, 0.0, 1.0);
  const cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64F);

  cv::Mat rvec;
  cv::Mat tvec;
  if (!cv::solvePnP(modelPoints, imagePoints, cameraMatrix, distCoeffs, rvec,
                    tvec, false, cv::SOLVEPNP_ITERATIVE)) {
    return pose;
  }

  cv::Mat rotMat;
  cv::Rodrigues(rvec, rotMat);
  const double r00 = rotMat.at<double>(0, 0);
  const double r10 = rotMat.at<double>(1, 0);
  const double r11 = rotMat.at<double>(1, 1);
  const double r12 = rotMat.at<double>(1, 2);
  const double r20 = rotMat.at<double>(2, 0);
  const double r21 = rotMat.at<double>(2, 1);
  const double r22 = rotMat.at<double>(2, 2);

  const double sy = std::sqrt(r00 * r00 + r10 * r10);
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  if (sy < 1e-6) {
    x = std::atan2(-r12, r11);
    y = std::atan2(-r20, sy);
    z = 0.0;
  } else {
    x = std::atan2(r21, r22);
    y = std::atan2(-r20, sy);
    z = std::atan2(r10, r00);
  }

  pose.pitch = static_cast<float>(x);
  pose.yaw = static_cast<float>(y);
  pose.roll = static_cast<float>(z);
  pose.valid = std::isfinite(pose.pitch) && std::isfinite(pose.yaw) &&
               std::isfinite(pose.roll);
  return pose;
}

int angleToSlot(float angle, int slots) {
  const float twoPi = 2.0f * static_cast<float>(CV_PI);
  float angleNorm = std::fmod(angle, twoPi);
  if (angleNorm < 0.0f) {
    angleNorm += twoPi;
  }
  int slot = static_cast<int>(std::floor(angleNorm / twoPi * slots));
  if (slot >= slots) {
    slot = slots - 1;
  }
  return slot;
}

PoseEstimate estimatePoseSlot(const FaceEmbedding::FaceData &data,
                              const cv::Size &frameSize, int slots,
                              float minMagnitude, float yawMax,
                              float pitchMax) {
  PoseEstimate estimate;
  if (data.landmarks.size() < 68) {
    return estimate;
  }

  const cv::Point2f leftEye = meanPoints(data.landmarks, 36, 41);
  const cv::Point2f rightEye = meanPoints(data.landmarks, 42, 47);
  const cv::Point2f nose = toPoint(data.landmarks[30]);
  const cv::Point2f mouthLeft = toPoint(data.landmarks[48]);
  const cv::Point2f mouthRight = toPoint(data.landmarks[54]);

  const cv::Point2f eyeMid = (leftEye + rightEye) * 0.5f;
  const cv::Point2f mouthMid = (mouthLeft + mouthRight) * 0.5f;

  const float distL = nose.x - leftEye.x;
  const float distR = rightEye.x - nose.x;
  const float yawRaw = (distR - distL) / (distR + distL + 1e-6f);

  const float pitchRaw = (nose.y - eyeMid.y) / (mouthMid.y - eyeMid.y + 1e-6f);
  float pitchNormFallback = (pitchRaw - 0.5f) / 0.25f;
  pitchNormFallback = clampf(pitchNormFallback, -1.0f, 1.0f);

  const float yawScreenFallback = clampf(-yawRaw, -1.0f, 1.0f);
  const float pitchScreenFallback = pitchNormFallback;

  float yawNorm = yawScreenFallback;
  float pitchNorm = pitchScreenFallback;

  const HeadPose pose = estimateHeadPose(data, frameSize);
  if (pose.valid) {
    float yawSolve = clampf(pose.yaw / yawMax, -1.0f, 1.0f);
    float pitchSolve = clampf(pose.pitch / pitchMax, -1.0f, 1.0f);
    yawSolve = -yawSolve;

    const float yawAlignThreshold = 0.12f;
    if (std::abs(yawScreenFallback) > yawAlignThreshold &&
        yawSolve * yawScreenFallback < 0.0f) {
      yawSolve = -yawSolve;
    }

    const float pitchAlignThreshold = 0.12f;
    if (std::abs(pitchScreenFallback) > pitchAlignThreshold &&
        pitchSolve * pitchScreenFallback < 0.0f) {
      pitchSolve = -pitchSolve;
    }

    yawNorm = yawSolve;
    pitchNorm = pitchSolve;
  }

  estimate.yawNorm = yawNorm;
  estimate.pitchNorm = pitchNorm;
  const float magnitude = std::sqrt(yawNorm * yawNorm + pitchNorm * pitchNorm);
  float angle = std::atan2(pitchNorm, yawNorm) - kRingStartAngle;
  if (!std::isfinite(angle)) {
    angle = 0.0f;
  }

  int slot = angleToSlot(angle, slots);
  if (magnitude < minMagnitude) {
    slot = 0;
    angle = 0.0f;
  }

  estimate.slot = slot;
  estimate.magnitude = magnitude;
  estimate.valid = true;
  return estimate;
}

std::string makeEmbeddingJson(const dlib::matrix<float, 0, 1> &embedding,
                              const PoseSummary &pose) {
  std::string json = "{\n  \"embedding\": [";
  for (long i = 0; i < embedding.size(); ++i) {
    if (i > 0) {
      json += ", ";
    }
    json += std::to_string(embedding(i));
  }
  json += "],\n  \"pose\": {\n";
  json += "    \"valid\": ";
  json += pose.valid ? "true" : "false";
  json += ",\n";
  json += "    \"yawNorm\": " + std::to_string(pose.yawNorm) + ",\n";
  json += "    \"pitchNorm\": " + std::to_string(pose.pitchNorm) + ",\n";
  json += "    \"magnitude\": " + std::to_string(pose.magnitude) + ",\n";
  json += "    \"slot\": " + std::to_string(pose.slot) + "\n";
  json += "  }\n}\n";
  return json;
}

} // namespace

int main(int argc, char **argv) {
  // Ensure logs go to stderr and keep stdout clean for JSON output.
  try {
    auto logger = spdlog::stderr_logger_mt("faceid_cli");
    spdlog::set_default_logger(logger);
  } catch (...) {
    // Ignore logger init errors; stdout must stay clean.
  }
  spdlog::set_level(spdlog::level::err);
  spdlog::flush_on(spdlog::level::err);

  if (argc < 2) {
    std::cerr << "Usage: faceid_cli <image_path>\n";
    return 2;
  }

  const std::filesystem::path imagePath = argv[1];
  if (!std::filesystem::exists(imagePath)) {
    std::cerr << "Image not found: " << imagePath << "\n";
    return 2;
  }

  const std::string arcface_path = "resources/arcface/arcface.onnx";
  const std::string dlib_path =
      "resources/dlib/dlib_face_recognition_resnet_model_v1.dat";
  const std::filesystem::path arcface_full =
      std::filesystem::current_path() / arcface_path;
  const std::string embedder_path =
      std::filesystem::exists(arcface_full) ? arcface_path : dlib_path;

  try {
    FaceEmbedding faceEmbedding(embedder_path);
    FaceRecognition faceRecognition("resources/dnn/deploy.prototxt",
                                    "resources/dnn/res10_300x300_ssd_iter_"
                                    "140000.caffemodel",
                                    faceEmbedding);

    cv::Mat image = cv::imread(imagePath.string());
    if (image.empty()) {
      std::cerr << "Failed to read image: " << imagePath << "\n";
      return 2;
    }

    const auto faces = faceRecognition.detectFaces(image);
    if (faces.empty()) {
      std::cerr << "No face detected\n";
      return 3;
    }

    const cv::Rect face = pickLargestFace(faces);
    FaceEmbedding::FaceData data;
    if (!faceEmbedding.getFaceData(image, face, data)) {
      std::cerr << "Failed to compute embedding\n";
      return 4;
    }

    const PoseEstimate pose = estimatePoseSlot(
        data, image.size(), 8, 0.18f, 0.55f, 0.45f);
    PoseSummary summary;
    summary.valid = pose.valid;
    summary.yawNorm = pose.yawNorm;
    summary.pitchNorm = pose.pitchNorm;
    summary.magnitude = pose.magnitude;
    summary.slot = pose.slot;

    std::cout << makeEmbeddingJson(data.embedding, summary);
    return 0;
  } catch (const std::exception &e) {
    spdlog::error("faceid_cli error: {}", e.what());
    std::cerr << e.what() << "\n";
    return 1;
  }
}

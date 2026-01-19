#include "FaceEmbedding/faceEmbedding.h"
#include "FaceMemory/faceMemory.h"
#include "FaceRecognition/faceRecognition.h"
#include "ImageProcessing/imageProcessing.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <dlib/matrix.h>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>

namespace {
cv::Rect pickLargestFace(const std::vector<cv::Rect> &faces) {
  if (faces.empty()) {
    return {};
  }
  return *std::max_element(
      faces.begin(), faces.end(),
      [](const cv::Rect &a, const cv::Rect &b) { return a.area() < b.area(); });
}

void drawOverlayLine(cv::Mat &frame, const std::string &text, int &line) {
  const int x = 10;
  const int y = 24 + line * 22;
  cv::putText(frame, text, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.6,
              cv::Scalar(255, 255, 255), 2);
  line++;
}

dlib::matrix<float, 0, 1>
averageEmbeddings(const std::vector<dlib::matrix<float, 0, 1>> &samples) {
  dlib::matrix<float, 0, 1> mean;
  if (samples.empty()) {
    return mean;
  }

  mean.set_size(samples[0].size());
  mean = 0;
  for (const auto &sample : samples) {
    mean += sample;
  }
  mean /= static_cast<float>(samples.size());
  return mean;
}

float maxDistanceFromMean(const std::vector<dlib::matrix<float, 0, 1>> &samples,
                          const dlib::matrix<float, 0, 1> &mean) {
  if (samples.empty() || mean.size() == 0) {
    return 0.0f;
  }

  float maxDistance = 0.0f;
  for (const auto &sample : samples) {
    maxDistance =
        std::max(maxDistance, static_cast<float>(dlib::length(sample - mean)));
  }
  return maxDistance;
}

double varianceOfLaplacian(const cv::Mat &gray) {
  cv::Mat lap;
  cv::Laplacian(gray, lap, CV_64F);
  cv::Scalar mean;
  cv::Scalar stddev;
  cv::meanStdDev(lap, mean, stddev);
  return stddev[0] * stddev[0];
}

struct FaceQuality {
  int width = 0;
  int height = 0;
  double blur = 0.0;
  bool ok = false;
};

FaceQuality evaluateFaceQuality(const cv::Mat &frame, const cv::Rect &rect,
                                int minSize, double blurThreshold) {
  FaceQuality quality;
  const cv::Rect bounded = rect & cv::Rect(0, 0, frame.cols, frame.rows);
  quality.width = bounded.width;
  quality.height = bounded.height;
  if (bounded.width < minSize || bounded.height < minSize) {
    quality.ok = false;
    return quality;
  }

  cv::Mat gray;
  cv::cvtColor(frame(bounded), gray, cv::COLOR_BGR2GRAY);
  quality.blur = varianceOfLaplacian(gray);
  quality.ok = quality.blur >= blurThreshold;
  return quality;
}

struct PoseBucket {
  std::vector<dlib::matrix<float, 0, 1>> samples;
  std::chrono::steady_clock::time_point lastCapture;
};

struct PoseEstimate {
  int slot = -1;
  float magnitude = 0.0f;
  float angle = 0.0f;
  bool valid = false;
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

PoseEstimate estimatePoseSlot(const FaceEmbedding::FaceData &data, int slots,
                              float minMagnitude) {
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
  const float yaw = (distR - distL) / (distR + distL + 1e-6f);

  const float pitch = (nose.y - eyeMid.y) / (mouthMid.y - eyeMid.y + 1e-6f);
  float pitchNorm = (pitch - 0.5f) / 0.25f;
  pitchNorm = clampf(pitchNorm, -1.0f, 1.0f);

  const float magnitude = std::sqrt(yaw * yaw + pitchNorm * pitchNorm);
  float angle = std::atan2(-pitchNorm, yaw);
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
  estimate.angle = angle;
  estimate.valid = true;
  return estimate;
}

bool allBucketsFilled(const std::vector<PoseBucket> &buckets,
                      int targetPerPose) {
  if (buckets.empty()) {
    return false;
  }
  return std::all_of(buckets.begin(), buckets.end(),
                     [targetPerPose](const PoseBucket &bucket) {
                       return static_cast<int>(bucket.samples.size()) >=
                              targetPerPose;
                     });
}

PoseBucket *bucketAt(std::vector<PoseBucket> &buckets, int slot) {
  if (slot < 0 || slot >= static_cast<int>(buckets.size())) {
    return nullptr;
  }
  return &buckets[slot];
}

int nextSlotFrom(const std::vector<PoseBucket> &buckets, int startSlot,
                 int targetPerSlot) {
  const int slots = static_cast<int>(buckets.size());
  if (slots == 0) {
    return -1;
  }
  for (int offset = 0; offset < slots; ++offset) {
    const int idx = (startSlot + offset) % slots;
    if (static_cast<int>(buckets[idx].samples.size()) < targetPerSlot) {
      return idx;
    }
  }
  return -1;
}

int collectedSamples(const std::vector<PoseBucket> &buckets) {
  int total = 0;
  for (const auto &bucket : buckets) {
    total += static_cast<int>(bucket.samples.size());
  }
  return total;
}

int filledSlots(const std::vector<PoseBucket> &buckets, int targetPerPose) {
  int total = 0;
  for (const auto &bucket : buckets) {
    if (static_cast<int>(bucket.samples.size()) >= targetPerPose) {
      total++;
    }
  }
  return total;
}

bool slotMatches(int slot, int active, int slots, int tolerance) {
  if (slots <= 0 || slot < 0 || active < 0) {
    return false;
  }
  const int diff = std::min((slot - active + slots) % slots,
                            (active - slot + slots) % slots);
  return diff <= tolerance;
}
} // namespace

cv::Point2f circlePoint(const cv::Point2f &center, float radius,
                        float angleRad) {
  return cv::Point2f(center.x + radius * std::cos(angleRad),
                     center.y + radius * std::sin(angleRad));
}

void drawFaceIdOverlay(cv::Mat &frame, const cv::Point &center, int radius,
                       const std::vector<PoseBucket> &buckets,
                       int targetPerSlot, int neededSlot) {
  if (buckets.empty()) {
    return;
  }

  const int slots = static_cast<int>(buckets.size());
  const int ticksPerSlot = 4;
  const int totalTicks = slots * ticksPerSlot;
  const float startAngle = -static_cast<float>(CV_PI) / 2.0f;
  const float step = 2.0f * static_cast<float>(CV_PI) / totalTicks;
  const cv::Scalar activeColor(0, 200, 0);
  const cv::Scalar pendingColor(170, 170, 170);
  const cv::Scalar highlightColor(230, 230, 230);

  for (int slot = 0; slot < slots; ++slot) {
    const bool filled =
        static_cast<int>(buckets[slot].samples.size()) >= targetPerSlot;
    cv::Scalar color = filled ? activeColor : pendingColor;
    if (slot == neededSlot && !filled) {
      color = highlightColor;
    }
    for (int t = 0; t < ticksPerSlot; ++t) {
      const int tickIndex = slot * ticksPerSlot + t;
      const float angle = startAngle + tickIndex * step;
      const cv::Point2f p1 =
          circlePoint(cv::Point2f(center), radius - 4.0f, angle);
      const cv::Point2f p2 =
          circlePoint(cv::Point2f(center), radius + 4.0f, angle);
      cv::line(frame, p1, p2, color, 2, cv::LINE_AA);
    }
  }

  const int faceRadius = static_cast<int>(radius * 0.35f);
  cv::circle(frame, center, faceRadius, cv::Scalar(200, 200, 200), 2,
             cv::LINE_AA);
  cv::circle(frame,
             cv::Point(center.x - faceRadius / 3, center.y - faceRadius / 5), 3,
             cv::Scalar(200, 200, 200), cv::FILLED, cv::LINE_AA);
  cv::circle(frame,
             cv::Point(center.x + faceRadius / 3, center.y - faceRadius / 5), 3,
             cv::Scalar(200, 200, 200), cv::FILLED, cv::LINE_AA);
  cv::ellipse(frame, cv::Point(center.x, center.y + faceRadius / 6),
              cv::Size(faceRadius / 3, faceRadius / 4), 0, 0, 180,
              cv::Scalar(200, 200, 200), 2, cv::LINE_AA);

  if (neededSlot >= 0 && neededSlot < slots) {
    const float slotAngle =
        startAngle +
        (neededSlot + 0.5f) * (2.0f * static_cast<float>(CV_PI) / slots);
    const cv::Point arrowEnd =
        circlePoint(cv::Point2f(center), radius * 0.55f, slotAngle);
    cv::arrowedLine(frame, center, arrowEnd, highlightColor, 2, cv::LINE_AA, 0,
                    0.2);
  }
}
// namespace

int main() {
  const std::string arcface_path = "bin/resources/arcface/arcface.onnx";
  const std::string dlib_path =
      "bin/resources/dlib/dlib_face_recognition_resnet_model_v1.dat";
  const std::filesystem::path arcface_full =
      std::filesystem::current_path() / arcface_path;
  const std::string embedder_path =
      std::filesystem::exists(arcface_full) ? arcface_path : dlib_path;
  FaceEmbedding faceEmbedding(embedder_path);
  FaceRecognition faceRecognition(
      "../resources/dnn/deploy.prototxt",
      "../resources/dnn/res10_300x300_ssd_iter_140000.caffemodel",
      faceEmbedding);

  const std::string face_db_path = "data/face_db.yml";
  FaceMemory faceMemory(0.5f);
  if (faceMemory.load(face_db_path)) {
    spdlog::info("Loaded faces: {}", faceMemory.size());
  } else {
    spdlog::info("Face database not found: {}", face_db_path);
  }
  if (faceMemory.embeddingDim() != 0 &&
      faceMemory.embeddingDim() != faceEmbedding.embeddingDim()) {
    spdlog::warn("Embedding dim mismatch (db={}, model={}). Clearing database.",
                 faceMemory.embeddingDim(), faceEmbedding.embeddingDim());
    faceMemory.clear();
    faceMemory.save(face_db_path);
  }

  cv::VideoCapture cap(0);
  cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
  cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
  if (!cap.isOpened()) {
    spdlog::error("Ошибка при открытии камеры!");
    return -1;
  }

  int frameCount = 0;
  bool showLandmarks = false;

  enum class UiMode { Idle, TypingName, Enrolling };
  UiMode uiMode = UiMode::Idle;
  std::string nameBuffer;
  std::string enrollName;
  std::vector<PoseBucket> poseBuckets;
  int activeSlot = -1;
  const int kPoseSlots = 12;
  const int kSamplesPerSlot = 3;
  const auto kEnrollInterval = std::chrono::milliseconds(350);
  const float kPoseMagnitudeMin = 0.18f;
  const int kSlotTolerance = 1;
  const int kMaxNameLength = 32;
  const float kThresholdMin = 0.35f;
  const float kThresholdMax = 0.75f;
  const float kThresholdStep = 0.02f;
  const int kMinFaceSize = 50;
  const double kBlurThreshold = 15.0;

  while (true) {
    cv::Mat frame;
    cap >> frame;
    if (frame.empty()) {
      spdlog::error("Empty frame captured!");
      break;
    }

    cv::flip(frame, frame, 1);

    const int key = cv::waitKey(1);
    if (uiMode == UiMode::TypingName) {
      if (key == 27) { // ESC
        uiMode = UiMode::Idle;
        nameBuffer.clear();
      } else if (key == 13 || key == 10) { // Enter
        if (nameBuffer.empty()) {
          spdlog::warn("Name is empty. Enrollment canceled.");
          uiMode = UiMode::Idle;
        } else {
          enrollName = nameBuffer;
          uiMode = UiMode::Enrolling;
          nameBuffer.clear();

          const auto now = std::chrono::steady_clock::now();
          poseBuckets.clear();
          poseBuckets.reserve(kPoseSlots);
          for (int i = 0; i < kPoseSlots; ++i) {
            poseBuckets.push_back({{}, now - kEnrollInterval});
          }
          activeSlot = 0;
          spdlog::info("Collecting samples for {}", enrollName);
        }
      } else if (key == 8 || key == 127) { // Backspace
        if (!nameBuffer.empty()) {
          nameBuffer.pop_back();
        }
      } else if (key >= 32 && key <= 126) {
        if (static_cast<int>(nameBuffer.size()) < kMaxNameLength) {
          nameBuffer.push_back(static_cast<char>(key));
        }
      }
    } else {
      if (key == 27 && uiMode == UiMode::Enrolling) { // ESC cancels enroll
        uiMode = UiMode::Idle;
        poseBuckets.clear();
        activeSlot = -1;
        spdlog::info("Enrollment canceled.");
      }

      switch (key) {
      case 'n': {
        uiMode = UiMode::TypingName;
        nameBuffer.clear();
        break;
      }
      case 'c': {
        faceMemory.clear();
        if (faceMemory.save(face_db_path)) {
          spdlog::info("Face database cleared.");
        } else {
          spdlog::warn("Failed to save empty database.");
        }
        break;
      }
      case 'r': {
        if (faceMemory.load(face_db_path)) {
          spdlog::info("Reloaded faces: {}", faceMemory.size());
        } else {
          spdlog::warn("Failed to reload database.");
        }
        break;
      }
      case 'q': {
        cap.release();
        cv::destroyAllWindows();
        return 0;
      }
      case '[': {
        const float next =
            std::max(kThresholdMin, faceMemory.threshold() - kThresholdStep);
        faceMemory.setThreshold(next);
        spdlog::info("Threshold set to {:.2f}", faceMemory.threshold());
        break;
      }
      case ']': {
        const float next =
            std::min(kThresholdMax, faceMemory.threshold() + kThresholdStep);
        faceMemory.setThreshold(next);
        spdlog::info("Threshold set to {:.2f}", faceMemory.threshold());
        break;
      }
      case 's': {
        ImageProcessing::saveFaceImage(frame, "data/snapshots", frameCount);
        break;
      }
      case 'm': {
        showLandmarks = !showLandmarks;
        spdlog::info("Landmarks: {}", showLandmarks ? "ON" : "OFF");
        break;
      }
      default:
        break;
      }
    }

    auto faceRects = faceRecognition.detectFaces(frame);

    std::string enrollHint;
    if (uiMode == UiMode::Enrolling) {
      if (faceRects.empty()) {
        enrollHint = "No face detected";
      } else if (faceRects.size() > 1) {
        enrollHint = "Only one face at a time";
      } else {
        const auto now = std::chrono::steady_clock::now();
        const cv::Rect target = pickLargestFace(faceRects);
        const FaceQuality quality =
            evaluateFaceQuality(frame, target, kMinFaceSize, kBlurThreshold);
        if (!quality.ok) {
          enrollHint = cv::format("Improve quality (%dx%d,%.0f)", quality.width,
                                  quality.height, quality.blur);
        } else {
          FaceEmbedding::FaceData data;
          if (!faceEmbedding.getFaceData(frame, target, data)) {
            enrollHint = "No landmarks";
          } else {
            const PoseEstimate pose =
                estimatePoseSlot(data, kPoseSlots, kPoseMagnitudeMin);
            if (!pose.valid) {
              enrollHint = "Hold face steady";
            } else {
              PoseBucket *bucket = bucketAt(poseBuckets, pose.slot);
              if (pose.magnitude < kPoseMagnitudeMin) {
                enrollHint = "Rotate head around the ring";
              } else {
                if (activeSlot >= 0 && bucket &&
                    slotMatches(pose.slot, activeSlot,
                                static_cast<int>(poseBuckets.size()),
                                kSlotTolerance) &&
                    static_cast<int>(bucket->samples.size()) <
                        kSamplesPerSlot &&
                    now - bucket->lastCapture >= kEnrollInterval) {
                  bucket->samples.push_back(data.embedding);
                  bucket->lastCapture = now;
                  if (static_cast<int>(bucket->samples.size()) >=
                      kSamplesPerSlot) {
                    activeSlot = nextSlotFrom(poseBuckets, activeSlot + 1,
                                              kSamplesPerSlot);
                  }
                }

                if (activeSlot >= 0) {
                  enrollHint = "Follow the ring clockwise";
                }
              }
            }
          }
        }
      }

      if (allBucketsFilled(poseBuckets, kSamplesPerSlot)) {
        bool anySaved = false;
        for (auto &bucket : poseBuckets) {
          const auto meanEmbedding = averageEmbeddings(bucket.samples);
          if (meanEmbedding.size() == faceEmbedding.embeddingDim()) {
            const float radius =
                maxDistanceFromMean(bucket.samples, meanEmbedding);
            if (faceMemory.add(enrollName, meanEmbedding, radius)) {
              anySaved = true;
            }
          }
        }

        if (anySaved && faceMemory.save(face_db_path)) {
          spdlog::info("Saved face '{}'. Total: {}", enrollName,
                       faceMemory.size());
        } else {
          spdlog::warn("Failed to save face {}", enrollName);
        }

        uiMode = UiMode::Idle;
        poseBuckets.clear();
        activeSlot = -1;
      }
    }

    if (uiMode == UiMode::Enrolling) {
      const int collected = collectedSamples(poseBuckets);
      const int total = static_cast<int>(poseBuckets.size()) * kSamplesPerSlot;

      cv::Point center(frame.cols / 2, frame.rows / 2);
      int radius = std::min(frame.cols, frame.rows) / 4;
      if (!faceRects.empty()) {
        const cv::Rect rect = pickLargestFace(faceRects);
        center = cv::Point(rect.x + rect.width / 2, rect.y + rect.height / 2);
        radius = std::max(std::min(rect.width, rect.height) / 2 + 20, 80);
      }
      drawFaceIdOverlay(frame, center, radius, poseBuckets, kSamplesPerSlot,
                        activeSlot);
    }

    for (const auto &rect : faceRects) {
      const FaceQuality quality =
          evaluateFaceQuality(frame, rect, kMinFaceSize, kBlurThreshold);
      const bool qualityOk = quality.ok;
      const bool requireQuality = (uiMode == UiMode::Enrolling);
      const bool allowEmbedding = qualityOk || !requireQuality;
      dlib::matrix<float, 0, 1> embedding;
      bool hasEmbedding = false;

      if (allowEmbedding) {
        if (showLandmarks) {
          FaceEmbedding::FaceData data;
          if (faceEmbedding.getFaceData(frame, rect, data)) {
            embedding = data.embedding;
            hasEmbedding = true;
            for (const auto &point : data.landmarks) {
              cv::circle(frame, cv::Point(point.x(), point.y()), 2,
                         cv::Scalar(0, 255, 255), -1);
            }
          }
        } else {
          if (faceEmbedding.getFaceDescriptor(frame, rect, embedding)) {
            hasEmbedding = true;
          }
        }
      }

      std::string label = "UNKNOWN";
      bool isMatch = false;
      if (uiMode == UiMode::Enrolling) {
        label = "ENROLLING";
      } else if (hasEmbedding && faceMemory.size() > 0) {
        const auto match = faceMemory.match(embedding);
        if (!match.name.empty()) {
          label = match.name + cv::format(" (%.2f)", match.distance);
          isMatch = true;
        } else if (std::isfinite(match.distance)) {
          label = "UNKNOWN" + cv::format(" (%.2f)", match.distance);
        } else {
          label = "UNKNOWN";
        }
      } else if (!hasEmbedding) {
        label = "NO EMB";
      } else if (faceMemory.size() == 0) {
        label = "NO DB";
      }

      const cv::Scalar color =
          isMatch ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
      cv::rectangle(frame, rect, color, 2);

      int baseline = 0;
      const cv::Size textSize =
          cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline);
      cv::Point textOrg(rect.x + 10, rect.y - textSize.height - 5);
      if (textOrg.y < 20) {
        textOrg.y = rect.y + 20;
      }

      cv::putText(frame, label, textOrg, cv::FONT_HERSHEY_SIMPLEX, 0.7, color,
                  2);
    }

    int line = 0;
    drawOverlayLine(frame,
                    "Keys: [n] new [c] clear [r] reload [m] landmarks [s] "
                    "snapshot [[/]] threshold [q] quit [ESC] cancel",
                    line);
    drawOverlayLine(frame,
                    "Threshold: " + cv::format("%.2f", faceMemory.threshold()),
                    line);
    drawOverlayLine(frame, "Known users: " + std::to_string(faceMemory.size()),
                    line);
    if (uiMode == UiMode::TypingName) {
      drawOverlayLine(frame, "Name: " + nameBuffer + "_", line);
      drawOverlayLine(frame, "Enter to start, Esc to cancel", line);
    } else if (uiMode == UiMode::Enrolling) {
      drawOverlayLine(frame, "Enrolling: " + enrollName, line);
      if (!poseBuckets.empty()) {
        const int segmentsFilled = filledSlots(poseBuckets, kSamplesPerSlot);
        const int totalSegments = static_cast<int>(poseBuckets.size());
        const int collected = collectedSamples(poseBuckets);
        const int totalSamples = totalSegments * kSamplesPerSlot;

        drawOverlayLine(frame,
                        "Coverage: " + std::to_string(segmentsFilled) + "/" +
                            std::to_string(totalSegments) + " segments",
                        line);
        drawOverlayLine(frame,
                        "Samples: " + std::to_string(collected) + "/" +
                            std::to_string(totalSamples),
                        line);
        if (activeSlot >= 0) {
          drawOverlayLine(frame,
                          "Target segment: " + std::to_string(activeSlot + 1) +
                              "/" + std::to_string(totalSegments),
                          line);
        }
      }
      if (!enrollHint.empty()) {
        drawOverlayLine(frame, enrollHint, line);
      }
    }

    cv::imshow("Webcam", frame);
    frameCount++;
  }

  cap.release();
  cv::destroyAllWindows();
  return 0;
}

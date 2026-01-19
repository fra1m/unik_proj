#include "faceMemory.h"

#include <algorithm>
#include <filesystem>
#include <limits>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

namespace {
dlib::matrix<float, 0, 1>
normalizeEmbedding(const dlib::matrix<float, 0, 1> &embedding) {
  const double norm = dlib::length(embedding);
  if (norm == 0.0) {
    return embedding;
  }
  return embedding / norm;
}

float l2Distance(const dlib::matrix<float, 0, 1> &a,
                 const dlib::matrix<float, 0, 1> &b) {
  return static_cast<float>(dlib::length(a - b));
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

cv::Mat dlibToCvMat(const dlib::matrix<float, 0, 1> &mat) {
  cv::Mat out(1, static_cast<int>(mat.size()), CV_32F);
  for (long i = 0; i < mat.size(); ++i) {
    out.at<float>(0, static_cast<int>(i)) = mat(i);
  }
  return out;
}
} // namespace

FaceMemory::FaceMemory(float threshold) : threshold_(threshold) {}

bool FaceMemory::load(const std::string &path) {
  records_.clear();
  embedding_dim_ = 0;
  if (!fs::exists(path)) {
    return false;
  }

  cv::FileStorage fs(path, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    return false;
  }

  cv::FileNode dimNode = fs["embedding_dim"];
  if (!dimNode.empty()) {
    dimNode >> embedding_dim_;
  }

  cv::FileNode thresholdNode = fs["threshold"];
  if (!thresholdNode.empty()) {
    thresholdNode >> threshold_;
  }

  cv::FileNode records = fs["records"];
  if (records.type() != cv::FileNode::SEQ) {
    return false;
  }

  for (const auto &item : records) {
    std::string name;
    item["name"] >> name;
    if (name.empty()) {
      continue;
    }

    Record record;
    record.name = name;

    cv::FileNode prototypesNode = item["prototypes"];
    if (prototypesNode.type() == cv::FileNode::SEQ) {
      for (const auto &protoNode : prototypesNode) {
        cv::Mat embedding;
        float radius = 0.0f;
        std::string pose;
        protoNode["embedding"] >> embedding;
        protoNode["radius"] >> radius;
        protoNode["pose"] >> pose;

        if (embedding.empty()) {
          continue;
        }

        dlib::matrix<float, 0, 1> dlibEmbedding = cvMatToDlib(embedding);
        if (dlibEmbedding.size() == 0) {
          continue;
        }
        if (embedding_dim_ == 0) {
          embedding_dim_ = static_cast<int>(dlibEmbedding.size());
        } else if (dlibEmbedding.size() != embedding_dim_) {
          continue;
        }

        record.prototypes.push_back(
            {normalizeEmbedding(dlibEmbedding), radius, pose});
      }
    } else {
      cv::Mat embedding;
      float radius = 0.0f;
      std::string pose;
      item["embedding"] >> embedding;
      item["radius"] >> radius;
      item["pose"] >> pose;

      if (!embedding.empty()) {
        dlib::matrix<float, 0, 1> dlibEmbedding = cvMatToDlib(embedding);
        if (dlibEmbedding.size() != 0) {
          if (embedding_dim_ == 0) {
            embedding_dim_ = static_cast<int>(dlibEmbedding.size());
          } else if (dlibEmbedding.size() != embedding_dim_) {
            continue;
          }
          record.prototypes.push_back(
              {normalizeEmbedding(dlibEmbedding), radius, pose});
        }
      }
    }

    if (!record.prototypes.empty()) {
      auto it = std::find_if(records_.begin(), records_.end(),
                             [&name](const Record &existing) {
                               return existing.name == name;
                             });
      if (it == records_.end()) {
        records_.push_back(record);
      } else {
        it->prototypes.insert(it->prototypes.end(),
                              record.prototypes.begin(),
                              record.prototypes.end());
      }
    }
  }

  return !records_.empty();
}

bool FaceMemory::save(const std::string &path) const {
  fs::path outPath(path);
  if (outPath.has_parent_path()) {
    fs::create_directories(outPath.parent_path());
  }

  cv::FileStorage fs(path, cv::FileStorage::WRITE);
  if (!fs.isOpened()) {
    return false;
  }

  fs << "embedding_dim" << embedding_dim_;
  fs << "threshold" << threshold_;
  fs << "records" << "[";
  for (const auto &record : records_) {
    fs << "{";
    fs << "name" << record.name;
    fs << "prototypes" << "[";
    for (const auto &proto : record.prototypes) {
      fs << "{";
      fs << "embedding" << dlibToCvMat(proto.embedding);
      fs << "radius" << proto.radius;
      if (!proto.pose.empty()) {
        fs << "pose" << proto.pose;
      }
      fs << "}";
    }
    fs << "]";
    fs << "}";
  }
  fs << "]";

  return true;
}

void FaceMemory::clear() {
  records_.clear();
  embedding_dim_ = 0;
}

size_t FaceMemory::size() const { return records_.size(); }

bool FaceMemory::add(const std::string &name,
                     const dlib::matrix<float, 0, 1> &embedding,
                     float radius, const std::string &pose) {
  if (embedding.size() == 0) {
    return false;
  }
  if (embedding_dim_ == 0) {
    embedding_dim_ = static_cast<int>(embedding.size());
  } else if (embedding.size() != embedding_dim_) {
    return false;
  }
  auto it = std::find_if(records_.begin(), records_.end(),
                         [&name](const Record &record) {
                           return record.name == name;
                         });
  Prototype proto{normalizeEmbedding(embedding), radius, pose};
  if (it == records_.end()) {
    Record record;
    record.name = name;
    record.prototypes.push_back(proto);
    records_.push_back(std::move(record));
  } else {
    it->prototypes.push_back(proto);
  }
  return true;
}

float FaceMemory::threshold() const { return threshold_; }

void FaceMemory::setThreshold(float value) { threshold_ = value; }

int FaceMemory::embeddingDim() const { return embedding_dim_; }

FaceMemory::Match
FaceMemory::match(const dlib::matrix<float, 0, 1> &embedding) const {
  Match best;
  best.distance = std::numeric_limits<float>::infinity();
  Match bestAny;
  bestAny.distance = std::numeric_limits<float>::infinity();

  if (records_.empty() || embedding.size() == 0) {
    return best;
  }
  if (embedding_dim_ != 0 && embedding.size() != embedding_dim_) {
    return best;
  }

  const auto query = normalizeEmbedding(embedding);
  for (const auto &record : records_) {
    for (const auto &proto : record.prototypes) {
      const float distance = l2Distance(query, proto.embedding);
      float allowed = threshold_;
      if (proto.radius > 0.0f) {
        const float relaxed = std::max(proto.radius + 0.15f, 0.35f);
        allowed = std::min(allowed, relaxed);
      }

      if (distance < bestAny.distance) {
        bestAny.name.clear();
        bestAny.distance = distance;
      }

      if (distance <= allowed && distance < best.distance) {
        best.name = record.name;
        best.distance = distance;
      }
    }
  }

  if (best.distance == std::numeric_limits<float>::infinity()) {
    return bestAny;
  }

  return best;
}

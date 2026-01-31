#pragma once

#include <dlib/matrix.h>
#include <string>
#include <vector>

class FaceMemory {
public:
  struct Match {
    std::string name;
    float distance;
  };

  explicit FaceMemory(float threshold = 0.5f);

  bool load(const std::string &path);
  bool save(const std::string &path) const;
  void clear();
  size_t size() const;

  bool add(const std::string &name, const dlib::matrix<float, 0, 1> &embedding,
           float radius, const std::string &pose = "");
  float threshold() const;
  void setThreshold(float value);
  int embeddingDim() const;
  Match match(const dlib::matrix<float, 0, 1> &embedding) const;

private:
  struct Prototype {
    dlib::matrix<float, 0, 1> embedding;
    float radius = 0.0f;
    std::string pose;
  };

  struct Record {
    std::string name;
    std::vector<Prototype> prototypes;
  };

  float threshold_;
  std::vector<Record> records_;
  int embedding_dim_ = 0;
};

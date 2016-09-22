// Copyright © 2016 Singulariti, Inc.
// All rights reserved.
// Created by Hao Chen (hao.chen@misingularity.io) on 6/5/2016

#include "tensorflow/cc/speech/pkg/dict.h"

namespace clm {

/* static */ const std::string WordsDictionary::kUnknown = "<unk>";
/* static */ const std::string WordsDictionary::kEndTag = "<eos>";

WordsDictionary::WordsDictionary(const std::vector<std::string>& v) {
  for (int i = 0; i < v.size(); i++) {
    id_to_word_.push_back(v[i]);
    word_to_id_.insert(std::pair<std::string, int>(v[i], i));
    if (v[i] == kUnknown) {
      kUnknown_id = i;
    }
    if (v[i] == kEndTag) {
      kEndTag_id = i;
    }
  }
}

std::string WordsDictionary::getWordById(int id) const {
  return id_to_word_[id];
}

int WordsDictionary::getIdByWord(const std::string& st) const {
  std::unordered_map<std::string, int>::const_iterator iter =
        word_to_id_.find(st);
  if (iter == word_to_id_.end()) {
    return kUnknown_id;
  } else {
    return iter->second;
  }
}

bool WordsDictionary::isUnknownId(int id) const {
  return id == kUnknown_id;
}

bool WordsDictionary::isEndTagId(int id) const {
  return id == kEndTag_id;
}

}  // namespace androidclm


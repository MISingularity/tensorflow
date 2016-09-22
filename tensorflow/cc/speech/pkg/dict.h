// Copyright © 2016 Singulariti, Inc.
// All rights reserved.
// Created by Hao Chen (hao.chen@misingularity.io) on 6/5/2016

#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace clm {

class WordsDictionary {
 public:
  WordsDictionary(const std::vector<std::string>&);

  std::string getWordById(int id) const;

  int getIdByWord(const std::string&) const;

  bool isUnknownId(int id) const;
  bool isEndTagId(int id) const;

 private:
  std::unordered_map<std::string, int> word_to_id_;
  std::vector<std::string> id_to_word_;

  static const std::string kUnknown;
  static const std::string kEndTag;

  int kUnknown_id;
  int kEndTag_id;
};

}  // namespace androidclm


// featbin/get-spkve-feat.cc

// Copyright 2014  Yajie Miao   Carnegie Mellon University

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Broadcast per-speaker/utterance features to frame-level\n"
        "Per-utterance by default, or per-speaker if utt2spk option provided\n"
        "Usage:get-spkvec-feat [options] (spkvec-rspecifier|spkvec-rxfilename) feats-rspecifier feats-wspecifier\n";

    ParseOptions po(usage);
    std::string utt2spk_rspecifier;
    po.Register("utt2spk", &utt2spk_rspecifier, "rspecifier for utterance to speaker map");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string spkvec_rspecifier = po.GetArg(1);
    std::string feat_rspecifier = po.GetArg(2);
    std::string feat_wspecifier = po.GetArg(3);

    SequentialBaseFloatMatrixReader feat_reader(feat_rspecifier);
    BaseFloatMatrixWriter feat_writer(feat_wspecifier);
    
    RandomAccessBaseFloatVectorReader spkvec_reader(spkvec_rspecifier);

    RandomAccessTokenReader utt2spk_reader(utt2spk_rspecifier);

    for (;!feat_reader.Done(); feat_reader.Next()) {
        std::string utt = feat_reader.Key();
        std::string utt_or_spk;
        if (utt2spk_rspecifier == "") utt_or_spk = utt;
        else {
          if (utt2spk_reader.HasKey(utt))
            utt_or_spk = utt2spk_reader.Value(utt);
          else {  // can't really recover from this error.
            KALDI_WARN << "Utt2spk map has no value for utterance "
                       << utt << ", producing no output for this utterance";
            continue;
          }
        }
        MatrixIndexT num_rows = feat_reader.Value().NumRows();
        const Vector<BaseFloat> &spkvec = spkvec_reader.Value(utt_or_spk);
        MatrixIndexT num_cols = spkvec.Dim();
        Matrix<BaseFloat> feat(num_rows, num_cols);
        for (MatrixIndexT r = 0; r < num_rows; ++r) {
          for (MatrixIndexT c = 0; c < num_cols; ++c) {
            feat(r,c) = spkvec(c);
          }
        }

        feat_writer.Write(utt, feat);
    }
    return 0; 
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}


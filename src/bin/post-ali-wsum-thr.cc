// bin/post-ali-wsum-thr.cc

// Copyright 2016  Chengzhu Yu
//           2013  Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
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
#include "hmm/posterior.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;  

    const char *usage =
        "Weighted sum of posterior probability vector and force alignment, weighted by scale paprameter "
        " post = scale*post + (1-scale)*post2 each for each utterance. post2 is zero vector with only aligned pdf has value of one \n"
        "Usage: post-weighted-sum <post1-rspecifier> <alignments-rspecifier> (<scale-rspecifier>|<scale>) <post-wspecifier>\n";
    
    ParseOptions po(usage); 

    BaseFloat min_post = 0.01;
    po.Register("min-post", &min_post, "Minimum posterior we will output (smaller "
                "ones are pruned).");
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }
      
    std::string post_rspecifier = po.GetArg(1),
                alignments_rspecifier =po.GetArg(2),
        scale_or_scale_rspecifier = po.GetArg(3),
        post_wspecifier = po.GetArg(4);

    double global_scale = 0.0;
    if (ClassifyRspecifier(scale_or_scale_rspecifier, NULL, NULL) == kNoRspecifier) {
      // treat second arg as a floating-point scale.
      if (!ConvertStringToReal(scale_or_scale_rspecifier, &global_scale))
        KALDI_ERR << "Bad third argument " << scale_or_scale_rspecifier
                  << " (expected scale or scale rspecifier)";
      scale_or_scale_rspecifier = ""; // So the archive won't be opened.
    }

    SequentialPosteriorReader posterior_reader(post_rspecifier);
    SequentialInt32VectorReader alignment_reader(alignments_rspecifier);
    RandomAccessBaseFloatReader scale_reader(scale_or_scale_rspecifier);
    PosteriorWriter posterior_writer(post_wspecifier); 

    int32 num_scaled = 0, num_no_scale = 0, num_no_post=0;  
   
    for (; !posterior_reader.Done(); posterior_reader.Next()) {
      std::string key = posterior_reader.Key();
      Posterior posterior = posterior_reader.Value();
      posterior_reader.FreeCurrent();

      // read alignment
      const std::vector<int32> &alignment = alignment_reader.Value();


      // Create an empty posterior object to be output after thretholding
      Posterior post_out(posterior.size());  
 
      if (scale_or_scale_rspecifier != "" && !scale_reader.HasKey(key)) {
        num_no_scale++;
      } else {
        BaseFloat post_scale = (scale_or_scale_rspecifier == "" ? global_scale
                                : scale_reader.Value(key));

             if (alignment_reader.Key() == key){
                 WeightedSumPosteriorAli(post_scale, &posterior, alignment, &post_out, min_post);
                 alignment_reader.Next();
                 posterior_writer.Write(key, post_out);
             } else{
                 posterior_writer.Write(key, posterior);
                 num_no_post++;
             }

        num_scaled++; 
      }
    }
    KALDI_LOG << "Done " << num_scaled << " posteriors;  " << num_no_post
              << " had no second alignments.";
    return (num_scaled != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


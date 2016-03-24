// bin/ali-to-post-map.cc

// Copyright 2016  Chengzhu Yu

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

int main(int argc, char *argv[]){
	try {

		using namespace kaldi;
		typedef kaldi::int32 int32;

		const char *usage =
		    "Train a mapping from forced aligned senone to a bunch of senones/pdfs that is the output of either DNN/GMM prediction "
		    "Usage: ali-to-post-map <alignments-rspecifier> <post-rspecifier> <map-rspecifier> <weight-rspecifier> <mapped_post-wspecifier>\n";

		BaseFloat min_post = 0.01;    
		ParseOptions po(usage);

		po.Register("min-post", &min_post, "Minimum posterior we will output (smaller "
                "ones are pruned).");

		
		po.Read(argc, argv);

		if (po.NumArgs() != 5) {
		  po.PrintUsage();
		  exit(1);
		}

		std::string alignments_rspecifier =po.GetArg(1),
					post_rspecifier = po.GetArg(2),
					map_rxfilename = po.GetArg(3),
					weigtht_rspecifier = po.GetArg(4),
					post_wspecifier = po.GetArg(5);


		SequentialPosteriorReader posterior_reader(post_rspecifier);
		SequentialInt32VectorReader alignment_reader(alignments_rspecifier);
		PosteriorWriter posterior_writer(post_wspecifier);

		// Read in trained mapping matrix, ali to posterior
		
		Matrix<BaseFloat> PostMap;
		{
		bool binary_in;
        Input ki(map_rxfilename, &binary_in);
        PostMap.Read(ki.Stream(), binary_in);
    	}

        // Read in weighting parameter, 
		BaseFloat w_org;
		ConvertStringToReal(weigtht_rspecifier, &w_org);

		int32 num_no_post = 0;
		for (; !posterior_reader.Done(); posterior_reader.Next()) {
		  std::string key = posterior_reader.Key();
		  Posterior posterior = posterior_reader.Value();
		  posterior_reader.FreeCurrent();

		  Posterior posterior_out(posterior.size());

		  // read second posterior
		  const std::vector<int32> &alignment = alignment_reader.Value();

		    if (alignment_reader.Key() == key){
		             AliPostMap( &posterior, alignment, &PostMap, w_org, min_post, &posterior_out );
		             alignment_reader.Next();
		             posterior_writer.Write(key, posterior_out);
		    } else{
		    		 posterior_writer.Write(key, posterior_out);
		             num_no_post++;

		    }
		    
		  }
		 
		KALDI_LOG << "Done " << ";" << " posteriors;  " << num_no_post
		          << " had no second alignments.";
  		
  		return 1;	
		}
	 	catch(const std::exception &e){
			std::cerr << e.what();
			return -1;
	}		


}

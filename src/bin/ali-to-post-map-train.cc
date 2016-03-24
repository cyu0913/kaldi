// bin/scale-post.cc

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
		    "Usage: ali-to-post-map-train <alignments-rspecifier> <post-rspecifier> <map-wspecifier> <ali-senone-number> <post-gmm-number>\n";

		ParseOptions po(usage);
		bool binary = false;

		po.Read(argc, argv);

		if (po.NumArgs() != 5) {
		  po.PrintUsage();
		  exit(1);
		}

		std::string alignments_rspecifier =po.GetArg(1),
					post_rspecifier = po.GetArg(2),
					aliSize_rspecifier = po.GetArg(3),
					postSize_rspecifier = po.GetArg(4),
					map_wspecifier = po.GetArg(5);


		SequentialPosteriorReader posterior_reader(post_rspecifier);
		SequentialInt32VectorReader alignment_reader(alignments_rspecifier);

		int32 ali_size;
		int32 post_size;
		ConvertStringToInteger(aliSize_rspecifier, &ali_size);
		ConvertStringToInteger(postSize_rspecifier, &post_size);

		Matrix<BaseFloat> map_sum(ali_size,post_size);			

		int32 frmNum=0;
		int32 tot_frmNum=0;
		int32 num_no_post=0;

		for (; !posterior_reader.Done(); posterior_reader.Next()) {
		  std::string key = posterior_reader.Key();
		  Posterior posterior = posterior_reader.Value();
		  posterior_reader.FreeCurrent();

		  // read second posterior
		  const std::vector<int32> &alignment = alignment_reader.Value();

		    if (alignment_reader.Key() == key){
		             AliPostMapSum( &posterior, alignment, &map_sum );
		             alignment_reader.Next();
		             frmNum = posterior.size();
		    } else{
		             num_no_post++;

		    }
			tot_frmNum += frmNum;
		    
		  }

		  {
		  	Output ko(map_wspecifier, binary);
		  	//WriteBasicType(ko.Stream(), binary, tot_frmNum);
		  	map_sum.Write(ko.Stream(),binary);
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

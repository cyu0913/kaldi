// ali-to-post-map-sum.cc

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

int main(int argc, char *argv[]) {
 try {
    typedef kaldi::int32 int32;
    using namespace kaldi;
    
    const char *usage =
        "Sum mapping matrix for mapping a new posterior from alignment \n"
        "Usage: ali-to-post-map-sum [options] <maps-in1> "
        "<maps-in2> ... <maps-inN> <maps-out>\n";

    bool binary = false;
    kaldi::ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    
    po.Read(argc, argv);

    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string maps_wxfilename = po.GetArg(po.NumArgs());

	Matrix<float> map_sum;

    for (int32 i = 1; i < po.NumArgs(); i++) {


        std::string maps_rxfilename = po.GetArg(i);
        KALDI_LOG << "Reading maps from " << maps_rxfilename;

        Matrix<BaseFloat> curr_map;

        {
        bool binary_in;
        Input ki(maps_rxfilename, &binary_in);
        curr_map.Read(ki.Stream(), binary_in);
    	}

        if (i==1){
        	map_sum.Resize(curr_map.NumRows(),curr_map.NumCols());
        }

		map_sum.AddMat(1.0, curr_map, kNoTrans);        

     }


     for (int32 i=0;i<map_sum.NumRows();i++){

     	BaseFloat tmp=0;

     	for(int32 j=0;j<map_sum.NumCols();j++){
     		tmp += map_sum(i,j);
     	}

     	if (tmp == 0){
     		tmp = 0.0001;
     	}

     	for(int32 j=0;j<map_sum.NumCols();j++){
     		map_sum(i,j) = map_sum(i,j)/tmp;
     		//map_sum(i,j) = map_sum(i,j);
     	}    	
     }

     {
     	Output ko(maps_wxfilename,binary);
     	map_sum.Write(ko.Stream(),binary);
     }

     return 0;
 }
 catch(const std::exception &e){
      std::cerr << e.what();
      return -1;
}
}









// featbin/transform-feats.cc

// Copyright 2009-2012  Microsoft Corporation
//                      Johns Hopkins University (author: Daniel Povey)

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
#include "matrix/kaldi-matrix.h"
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "transform/regtree-fmllr-diag-gmm.h"
#include "gmm/am-diag-gmm.h"
#include "fst/fstlib.h"

int main(int argc, char *argv[]) {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Apply transform (e.g. LDA; HLDA; fMLLR/CMLLR; MLLT/STC)\n"
        "Linear transform if transform-num-cols == feature-dim, affine if\n"
        "transform-num-cols == feature-dim+1 (->append 1.0 to features)\n"
        "Per-utterance by default, or per-speaker if utt2spk option provided\n"
        "Global if transform-rxfilename provided.\n"
        "Usage: transform-feats-regtree [options] (<transform-rspecifier>|<transform-rxfilename>) <feats-rspecifier> <feats-wspecifier>\n"
        "See also: transform-vec, copy-feats, compose-transforms\n";
    
    ParseOptions po(usage);
    std::string utt2spk_rspecifier;
    po.Register("utt2spk", &utt2spk_rspecifier, "rspecifier for utterance to speaker map");

    po.Read(argc, argv);

    if (po.NumArgs() != 6) {
      po.PrintUsage();
      exit(1);
    }

    std::string feat_rspecifier = po.GetArg(1);
    std::string regtree_filename = po.GetArg(2);
    std::string xforms_rspecifier = po.GetArg(3);
    std::string alignments_rspecifier = po.GetArg(4);
    std::string model_filename = po.GetArg(5);
    std::string feat_wspecifier = po.GetArg(6);

    //TransitionModel trans_model;
    //ReadKaldiObject(model_filename, &trans_model);
    AmDiagGmm am_gmm;
    TransitionModel trans_model;
    {
        bool binary;
        Input ki(model_filename, &binary);
        trans_model.Read(ki.Stream(), binary);
        am_gmm.Read(ki.Stream(), binary);
    }

    RegressionTree regtree;
    {   
      bool binary_read;
      Input in(regtree_filename, &binary_read);
      regtree.Read(in.Stream(), binary_read, am_gmm);
    }   

    SequentialBaseFloatMatrixReader feat_reader(feat_rspecifier);
    BaseFloatMatrixWriter feat_writer(feat_wspecifier);
    RandomAccessTokenReader utt2spk_reader(utt2spk_rspecifier);
    RandomAccessRegtreeFmllrDiagGmmReader fmllr_reader(xforms_rspecifier);
    RandomAccessInt32VectorReader alignments_reader(alignments_rspecifier);

    for (;!feat_reader.Done(); feat_reader.Next()){
        std::string utt = feat_reader.Key();
        Matrix<BaseFloat> features(feat_reader.Value());
        Matrix<BaseFloat> features_out(features.NumRows(),features.NumCols());
        std::vector<int32> alignment = alignments_reader.Value(utt);

        std::string utt_or_spk = utt2spk_reader.Value(utt);
        RegtreeFmllrDiagGmm fmllr(fmllr_reader.Value(utt_or_spk));

        std::cout << fmllr.NumRegClasses() << std::endl;

        int32 NumFrame = features.NumRows();

        for (int32 frame = 0; frame < NumFrame; frame++){
             const VectorBase<BaseFloat> &feat =  features.Row(frame);
             std::vector< Vector<BaseFloat> > xformed_feat;
             fmllr.TransformFeature(feat, &xformed_feat);
                
             int32 PdfId = trans_model.TransitionIdToPdf(alignment[frame]);
             const DiagGmm &gmm = am_gmm.GetPdf(PdfId);

             unordered_map<int32, int32> hashes;
             for (int32 regid = 0; regid < fmllr.NumRegClasses(); regid++) hashes[regid] = 0;

             if (gmm.NumGauss() == 0) return -1;

             for (int32 mix = 0; mix < gmm.NumGauss(); mix++)
             {
                int32 baseclass = regtree.Gauss2BaseclassId(PdfId, mix);
                int32 regid = fmllr.Base2RegClass(baseclass);
                hashes[regid]++;
             }
             
             int32 regclass = 0;
             for (int32 regid = 0; regid < fmllr.NumRegClasses(); regid++) 
             {
                regclass = (hashes[regid] > hashes[regclass]) ? regid : regclass;
                //KALDI_LOG << hashes[regid] ;
             }  


             //KALDI_LOG << regclass << " Majority " ;
    
             Vector<BaseFloat> tmp_xformed_feat_perframe  = xformed_feat[regclass];

             features_out.CopyRowFromVec(tmp_xformed_feat_perframe, frame);
             //KALDI_LOG << "done uttrance " << utt << " frame " << frame;
             //KALDI_LOG << "input feature " << feat;
             //KALDI_LOG << "transformed feature " << tmp_xformed_feat_perframe;
         }
         KALDI_LOG << "done uttrance " << utt ;

         feat_writer.Write(utt, features_out);
     }

     return 0;
}




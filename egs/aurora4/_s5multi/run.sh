#!/bin/bash

log_start(){
  echo "#####################################################################"
  echo "Spawning *** $1 *** on" `date` `hostname`
  echo ---------------------------------------------------------------------
}

log_end(){
  echo ---------------------------------------------------------------------
  echo "Done *** $1 *** on" `date` `hostname` 
  echo "#####################################################################"
}

rm -f ./path.sh; cp ./path.cluster.sh ./path.sh;

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. ./path.sh ## update system path


#-------- Change the path of features stored in kaldi for using in different machine or directory------#
run_scp_edit(){
    curr=`pwd`

    dest1=/data/work2/chengzhu/work/Toolhouse/kaldi-trunk-4ntt/egs/aurora4/s5

    sed -i "s#$dest1#$curr#" mfcc/*.scp

    sed -i "s#$dest1#$curr#" fbank/*.scp

    rm -rf data/*/split* data-fbank/*/split*

    sed -i "s#$dest1#$curr#" data/*/feats.scp

    sed -i "s#$dest1#$curr#" data-fbank/*/feats.scp

    sed -i "s#$dest1#$curr#" data/*/cmvn.scp

    sed -i "s#$dest1#$curr#" data-fbank/*/cmvn.scp
}
#run_scp_edit

#--------------------------------------GMM-HMM SYSTEM-------------------------------------------------------------------------
run_gmm_hmm(){
    log_start "Training and Decoding GMM-HMM system"

    steps/train_mono_uttcmvn.sh --boost-silence 1.25 --nj 80 --cmd "$train_cmd" \
    data/train_si84_multi data/lang exp/mono0a_multi || exit 1;
    
    steps/align_si_uttcmvn.sh --boost-silence 1.25 --nj 50 --cmd "$train_cmd" \
       data/train_si84_multi data/lang exp/mono0a_multi exp/mono0a_multi_ali || exit 1;
    
    steps/train_deltas_uttcmvn.sh --boost-silence 1.25 --cmd "$train_cmd"  \
        4500 55000 data/train_si84_multi data/lang exp/mono0a_multi_ali exp/tri1_multi || exit 1;
    
    steps/align_si_uttcmvn.sh --nj 80 --cmd "$train_cmd"  \
       data/train_si84_multi data/lang exp/tri1_multi exp/tri1_multi_ali_si84 || exit 1;

    steps/train_deltas_uttcmvn.sh  --cmd "$train_cmd" \
       4500 55000 data/train_si84_multi data/lang exp/tri1_multi_ali_si84 exp/tri2a_multi || exit 1;

    steps/align_si_uttcmvn.sh --nj 80 --cmd "$train_cmd"  \
       data/train_si84_multi data/lang exp/tri2a_multi exp/tri2a_multi_ali_si84 || exit 1;
    
    steps/train_lda_mllt_uttcmvn.sh --cmd "$train_cmd"  \
        --splice-opts "--left-context=3 --right-context=3" \
       4500 55000 data/train_si84_multi data/lang exp/tri2a_multi_ali_si84 exp/tri2b_multi || exit 1;
    
    #Trigram
    utils/mkgraph.sh data/lang_test_tgpr_5k exp/tri2b_multi exp/tri2b_multi/graph_tgpr_5k || exit 1;
    steps/decode_uttcmvn.sh --nj 80 --cmd "$decode_cmd" \
        exp/tri2b_multi/graph_tgpr_5k data/test_eval92 exp/tri2b_multi/decode_tgpr_5k_eval92 || exit 1;
    
     #Bigram
    utils/mkgraph.sh data/lang_test_bg_5k exp/tri2b_multi exp/tri2b_multi/graph_bg_5k || exit 1;
    steps/decode_uttcmvn.sh --nj 80 --cmd "$decode_cmd" \
        exp/tri2b_multi/graph_bg_5k data/test_eval92 exp/tri2b_multi/decode_bg_5k_eval92 || exit 1;
    
    steps/align_si_uttcmvn.sh  --nj 40 --cmd "$train_cmd" \
      data/train_si84_multi data/lang exp/tri2b_multi exp/tri2b_multi_ali_si84  || exit 1;
    
    steps/align_si_uttcmvn.sh  --nj 80 --cmd "$train_cmd" \
      data/dev_1206 data/lang exp/tri2b_multi exp/tri2b_multi_ali_dev_1206 || exit 1;

    steps/align_si_uttcmvn.sh  --nj 80 --cmd "$train_cmd" \
      data/test_eval92 data/lang exp/tri2b_multi exp/tri2b_multi_ali_eval92  || exit 1;

    log_end "Training and Decoding GMM-HMM system"
}
#run_gmm_hmm

    steps/align_si_uttcmvn.sh  --boost_silence 1.5 --nj 40 --cmd "$train_cmd" \
      data/train_si84_multi data/lang exp/tri2b_multi exp/tri2b_multi_ali_si84  || exit 1;
    
    steps/align_si_uttcmvn.sh  --boost_silence 1.5 --nj 80 --cmd "$train_cmd" \
      data/dev_1206 data/lang exp/tri2b_multi exp/tri2b_multi_ali_dev_1206 || exit 1;

    steps/align_si_uttcmvn.sh  --boost_silence 1.5 --nj 80 --cmd "$train_cmd" \
      data/test_eval92 data/lang exp/tri2b_multi exp/tri2b_multi_ali_eval92  || exit 1;




##----------------------------------------DNN-HMM SYSTEM---------------------------------------------------------------------------
rm -f ./path.sh; cp ./path.gpu.sh ./path.sh;
#. ./path.sh

run_rbm_pretrain(){
    log_start "RBM pretraining"

    dir=exp/tri3a_dnn_pretrain
    $cuda_cmd $dir/_pretrain_dbn.log \
      steps/nnet/pretrain_dbn_with_uttcmvn.sh --delta-opts '--delta-window=3 --delta-order=2' --nn-depth 5 --rbm-iter 3 --skip_cuda_check true --copy-feats true data-fbank/train_si84_multi $dir

    log_end "RBM pretraining"
}
#run_rbm_pretrain

run_fine_tune(){
    log_start "DNN Fintuning"

    dir=exp/tri3a_dnn
    ali=exp/tri2b_multi_ali_si84
    ali_dev=exp/tri2b_multi_ali_dev_1206
    feature_transform=exp/tri3a_dnn_pretrain/final.feature_transform
    dbn=exp/tri3a_dnn_pretrain/5.dbn
    
    $cuda_cmd $dir/_train_nnet.log \
      steps/nnet/train_with_uttcmvn.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 --skip_cuda_check true \
      data-fbank/train_si84_multi data-fbank/dev_1206 data/lang $ali $ali_dev $dir || exit 1;

    log_end "DNN Fintuning"
}
#run_fine_tune

run_dnn_decode(){
    log_start "DNN Decoding"

    rm -f ./path.sh; cp ./path.cluster.sh ./path.sh;

    dir=exp/tri3a_dnn

    #utils/mkgraph.sh data/lang_test_bg_5k exp/tri3a_dnn exp/tri3a_dnn/graph_bg_5k || exit 1;
    #steps/nnet/decode_uttcmvn.sh --nj 80 --cmd "$train_cmd" \
    #  --acwt 0.06 --config conf/decode_dnn.config  --beam 15.0 --lattice-beam 9.0 --max-active 15000 \
    #  $dir/graph_bg_5k data-fbank/test_eval92 $dir/decode_bg_5k_eval92 || exit 1;

    utils/mkgraph.sh data/lang_test_tgpr_5k exp/tri3a_dnn exp/tri3a_dnn/graph_tgpr_5k || exit 1;
    steps/nnet/decode_uttcmvn.sh --nj 80 --cmd "$train_cmd" \
      --acwt 0.10 --config conf/decode_dnn.config \
      $dir/graph_tgpr_5k data-fbank/test_eval92 $dir/decode_tgpr_5k_eval92 || exit 1;
    
    srcdir=exp/tri3a_dnn
    steps/nnet/align_uttcmvn.sh --nj 80 --cmd "$train_cmd" \
      data-fbank/train_si84_multi data/lang $srcdir ${srcdir}_ali_si84_multi || exit 1;

    log_end "DNN Decoding"
}
#run_dnn_decode


#-----------------------------DNN SEQUENTIAL TRAINING -----------------------------------------------#
run_dnn_seq(){
    log_start "DNN Sequential Training"

    dir=exp/tri3a_dnn_smbr
    srcdir=exp/tri3a_dnn
    acwt=0.10
    
    #-- First we need to generate lattices and alignments:
    {
    steps/nnet/make_denlats_uttcmvn.sh --nj 50 --cmd "$gpucpu_cmd" \
      --config conf/decode_dnn.config --acwt $acwt \
      data-fbank/train_si84_multi data/lang $srcdir ${srcdir}_denlats_si284  || exit 1;
    }
    
    #-- Now we re-train the hybrid by single iteration of sMBR 
    {
    steps/nnet/train_mpe_uttcmvn.sh --cmd "$cuda_cmd" --skip_cuda_check true --num-iters 1 --acwt $acwt --do-smbr true \
       data-fbank/train_si84_multi data/lang $srcdir \
      ${srcdir}_ali_si84_multi ${srcdir}_denlats_si284 $dir || exit 1
    }
    
    #-- Decode after single iteration of sMBR
    {
    for ITER in 1; do
      steps/nnet/decode_uttcmvn.sh --nj 40 --cmd "$gpucpu_cmd" --config conf/decode_dnn.config \
        --nnet $dir/${ITER}.nnet --acwt $acwt \
        exp/tri3a_dnn/graph_tgpr_5k data-fbank/test_eval92 $dir/decode_tgpr_5k_eval92${ITER} || exit 1
    
      steps/nnet/decode_uttcmvn.sh --nj 40 --cmd "$gpucpu_cmd" --config conf/decode_dnn.config \
        --nnet $dir/${ITER}.nnet --acwt $acwt \
        exp/tri3a_dnn/graph_bg_5k data-fbank/test_eval92 $dir/decode_bg_5k_eval92${ITER} || exit 1
    done
    }
    
    #-- Re-generate lattices and run several more iterations of sMBR
    
    dir=exp/tri3a_dnn_smbr_iter1-lats
    srcdir=exp/tri3a_dnn_smbr
    acwt=0.10
    
    #-- First we need to generate lattices and alignments:
    {
    steps/nnet/align_uttcmvn.sh --nj 40 --cmd "$gpucpu_cmd" \
      data-fbank/train_si84_multi data/lang $srcdir ${srcdir}_ali_si84_multi || exit 1;
    steps/nnet/make_denlats_uttcmvn.sh --nj 40 --cmd "$gpucpu_cmd" \
      --config conf/decode_dnn.config --acwt $acwt \
      data-fbank/train_si84_multi data/lang $srcdir ${srcdir}_denlats_si284  || exit 1;
    }
    
    #-- Now we re-train the hybrid by several iterations of sMBR 
    {
    steps/nnet/train_mpe_uttcmvn.sh --cmd "$cuda_cmd" --skip_cuda_check true --num-iters 4 --acwt $acwt --do-smbr true \
      data-fbank/train_si84_multi data/lang $srcdir \
      ${srcdir}_ali_si84_multi ${srcdir}_denlats_si284 $dir
    }
    
    #-- Final Decode
    {
    for ITER in 1 2 3 4; do
      steps/nnet/decode_uttcmvn.sh --nj 40 --cmd "$gpucpu_cmd" --config conf/decode_dnn.config \
        --nnet $dir/${ITER}.nnet --acwt $acwt \
        exp/tri3a_dnn/graph_tgpr_5k data-fbank/test_eval92 $dir/decode_tgpr_5k_eval92${ITER} || exit 1
    
      steps/nnet/decode_uttcmvn.sh --nj 40 --cmd "$gpucpu_cmd" --config conf/decode_dnn.config \
        --nnet $dir/${ITER}.nnet --acwt $acwt \
        exp/tri3a_dnn/graph_bg_5k data-fbank/test_eval92 $dir/decode_bg_5k_eval92${ITER} || exit 1
    done
    }

    log_end "DNN Sequential Training"
}
#run_dnn_seq


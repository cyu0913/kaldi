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

. cmd.sh
. path.sh

set -e # exit on error

run_mfcc(){
    mfccdir=mfcc
    for x in train eval2000; do
      steps/make_mfcc.sh --nj 80 --cmd "$train_cmd" \
        data/$x exp/make_mfcc/$x $mfccdir
      steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir 
      utils/fix_data_dir.sh data/$x
    done
}
#run_mfcc

run_data_split(){

    # Use the first 4k sentences as dev set.  Note: when we trained the LM, we used
    # the 1st 10k sentences as dev set, so the 1st 4k won't have been used in the
    # LM training data.   However, they will be in the lexicon, plus speakers
    # may overlap, so it's still not quite equivalent to a test set.
    utils/subset_data_dir.sh --first data/train 4000 data/train_dev || exit 1;# 5hr 6min
    n=$[`cat data/train/wav.scp | wc -l` - 4000]
    utils/subset_data_dir.sh --last data/train $n data/train_nodev || exit 1;
    
    # Now-- there are 260k utterances (313hr 23min), and we want to start the 
    # monophone training on relatively short utterances (easier to align), but not 
    # only the shortest ones (mostly uh-huh).  So take the 100k shortest ones;
    # remove most of the repeated utterances (these are the uh-huh type ones), and 
    # then take 10k random utterances from those (about 4hr 40mins)
    utils/subset_data_dir.sh --shortest data/train_nodev 100000 data/train_100kshort || exit 1;
    utils/subset_data_dir.sh data/train_100kshort 30000 data/train_30kshort || exit 1;
    
    ## Take the first 100k utterances (just under half the data); we'll use
    ## this for later stages of training.
    utils/subset_data_dir.sh --first data/train_nodev 100000 data/train_100k
    local/remove_dup_utts.sh 200 data/train_100k data/train_100k_nodup  # 110hr

    ## Finally, the full training set:
    local/remove_dup_utts.sh 300 data/train_nodev data/train_nodup  # 286hr
}
#run_data_split;

run_mono(){
    steps/train_mono.sh --nj 80 --cmd "$train_cmd" \
      data/train_30kshort data/lang exp/mono || exit 1;
}
#run_mono

run_tri1(){
    steps/align_si.sh --nj 80 --cmd "$train_cmd" \
      data/train_100k_nodup data/lang exp/mono exp/mono_ali || exit 1; 

    steps/train_deltas.sh --cmd "$train_cmd" \
      3200 30000 data/train_100k_nodup data/lang exp/mono_ali exp/tri1 || exit 1; 
}
#run_tri1

decode_tri1(){
    graph_dir=exp/tri1/graph_nosp_sw1_tg
    $train_cmd $graph_dir/mkgraph.log \
        utils/mkgraph.sh data/lang_test exp/tri1 $graph_dir
    steps/decode_si.sh --nj 80 --cmd "$decode_cmd" --config conf/decode.config \
        $graph_dir data/eval2000 exp/tri1/decode_eval2000_nosp_sw1_tg
}
#decode_tri1

run_tri2(){
    steps/align_si.sh --nj 80 --cmd "$train_cmd" \
       data/train_100k_nodup data/lang exp/tri1 exp/tri1_ali || exit 1;
    
    steps/train_deltas.sh --cmd "$train_cmd" \
       4000 70000 data/train_100k_nodup data/lang exp/tri1_ali exp/tri2 || exit 1;
}
#run_tri2

decode_tri2(){
    graph_dir=exp/tri2/graph_nosp_sw1_tg
    $train_cmd $graph_dir/mkgraph.log \
      utils/mkgraph.sh data/lang_test exp/tri2 $graph_dir
    steps/decode.sh --nj 80 --cmd "$decode_cmd" --config conf/decode.config \
      $graph_dir data/eval2000 exp/tri2/decode_eval2000_nosp_sw1_tg
  
}
#decode_tri2


run_tri3(){
    ## The 100k_nodup data is used in neural net training.
    steps/align_si.sh --nj 80 --cmd "$train_cmd" \
      data/train_100k_nodup data/lang exp/tri2 exp/tri2_ali_100k_nodup
    
    ## From now, we start using all of the data (except some duplicates of common
    ## utterances, which don't really contribute much).
    steps/align_si.sh --nj 80 --cmd "$train_cmd" \
      data/train_nodup data/lang exp/tri2 exp/tri2_ali_nodup 
    #
    ## Do another iteration of LDA+MLLT training, on all the data.
    steps/train_lda_mllt.sh --cmd "$train_cmd" \
      4000 70000 data/train_nodup data/lang exp/tri2_ali_nodup exp/tri3 
}
#run_tri3

decode_tri3(){
    graph_dir=exp/tri3/graph_nosp_sw1_tg
    $train_cmd $graph_dir/mkgraph.log \
        utils/mkgraph.sh data/lang_test exp/tri3 $graph_dir
    steps/decode.sh --nj 80 --cmd "$decode_cmd" --config conf/decode.config \
        $graph_dir data/eval2000 exp/tri3/decode_eval2000_nosp_sw1_tg
}
#decode_tri3

run_tri4(){
    steps/align_fmllr.sh --nj 80 --cmd "$train_cmd" \
      data/train_nodup data/lang exp/tri3 exp/tri3_ali_nodup 
    
    steps/train_sat.sh  --cmd "$train_cmd" \
      4199 80000 data/train_nodup data/lang exp/tri3_ali_nodup exp/tri4
      #11500 200000 data/train_nodup data/lang exp/tri3_ali_nodup exp/tri4
    }
#run_tri4

decode_tri4(){
  graph_dir=exp/tri4/graph_sw1_tg
  $train_cmd $graph_dir/mkgraph.log \
    utils/mkgraph.sh data/lang_test exp/tri4 $graph_dir
  steps/decode_fmllr.sh --nj 80 --cmd "$decode_cmd" \
    --config conf/decode.config \
    $graph_dir data/eval2000 exp/tri4/decode_eval2000_sw1_tg.text

    steps/align_fmllr.sh --nj 80 --cmd "$train_cmd" \
      data/train_nodup data/lang exp/tri4 exp/tri4_ali_nodup

}
#decode_tri4

store_fmllr_feat(){
   # Store fMLLR features, so we can train on them easily,

   # eval2000
   gmmdir=exp/tri4
   data_fmllr=exp/data-fmllr-tri4
   has_fisher=false
   dir=$data_fmllr/eval2000

   steps/nnet/make_fmllr_feats.sh --nj 80 --cmd "$train_cmd" \
     --transform-dir $gmmdir/decode_eval2000_sw1_tg.text \
     $dir data/eval2000 $gmmdir $dir/log $dir/data || exit 1

   # train
   dir=$data_fmllr/train_nodup
   steps/nnet/make_fmllr_feats.sh --nj 80 --cmd "$train_cmd" \
      --transform-dir ${gmmdir}_ali_nodup \
      $dir data/train_nodup $gmmdir $dir/log $dir/data || exit 1
   
   #split the data : 90% train 10% cross-validation (held-out)
   utils/subset_data_dir_tr_cv.sh $dir ${dir}_tr90 ${dir}_cv10 || exit 1
}
#store_fmllr_feat

#===================== DNN Training ===========================#
rm -f ./path.sh; cp ./path.gpu.sh ./path.sh;

gmmdir=exp/tri4
data_fmllr=exp/data-fmllr-tri4
has_fisher=false

run_dnn_pretrain(){
    log_start "DNN Pretraining"

    dir=exp/dnn5b_pretrain-dbn
    (tail --pid=$$ -F $dir/log/pretrain_dbn.log 2>/dev/null)& # forward log
    $cuda_cmd $dir/log/pretrain_dbn.log \
      steps/nnet/pretrain_dbn.sh --rbm-iter 1 $data_fmllr/train_nodup $dir || exit 1;
    
    log_end "DNN Pretraining"
}
run_dnn_pretrain

dir=exp/dnn5b_pretrain-dbn_dnn
run_dnn_ftn(){
    log_start "DNN Finetuning"

    ali=${gmmdir}_ali_nodup
    feature_transform=exp/dnn5b_pretrain-dbn/final.feature_transform
    dbn=exp/dnn5b_pretrain-dbn/6.dbn
    (tail --pid=$$ -F $dir/log/train_nnet.log 2>/dev/null)& # forward log
    # Train
    $cuda_cmd $dir/log/train_nnet.log \
      steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
      $data_fmllr/train_nodup_tr90 $data_fmllr/train_nodup_cv10 data/lang $ali $ali $dir || exit 1;

   log_end "DNN Finetuning"
}
run_dnn_ftn


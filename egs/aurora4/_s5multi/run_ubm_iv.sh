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

ubmdim=2048
ivdim=200
run_iv_extract(){
   log_start "Ivector Extraction"

   sid/train_diag_ubm.sh --parallel-opts "" --nj 80 --cmd "$train_cmd" data/train_si84_multi ${ubmdim} \
       exp/diag_ubm_${ubmdim} || exit 1;

   sid/train_full_ubm.sh --nj 80 --cmd "$train_cmd" data/train_si84_multi \
       exp/diag_ubm_${ubmdim} exp/full_ubm_${ubmdim} || exit 1;

   sid/train_ivector_extractor.sh --nj 80 --cmd "$train_cmd" --num-gselect 20 \
      --ivector-dim $ivdim --num-iters 10 exp/full_ubm_${ubmdim}/final.ubm data/train_si84_multi \
      exp/extractor_${ubmdim} || exit 1;

   sid/extract_ivectors.sh --cmd "$train_cmd" --nj 80 --num-gselect 1 \
       exp/extractor_${ubmdim} data/test_eval92 data/test_eval92.iv || exit 1;

   sid/extract_ivectors.sh --cmd "$train_cmd" --nj 80 --num-gselect 1 \ 
        exp/extractor_${ubmdim} data/train_si84_multi data/train_si84_multi.iv || exit 1;


   log_end "Ivector Extraction"
}
#run_iv_extract





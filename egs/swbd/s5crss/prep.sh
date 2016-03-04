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
has_fisher=true

run_prep_trans(){
   # check if data/local/swb_ms98_transcriptions exists.
   # if not copy it from /scratch2/cxy110530/SWB_KALDI_COPY/swb_kaldi/data/local/train/swb_ms98_transcriptions
   echo "Skip this step, Check the comment"
}
#run_prep_trans

run_prep(){
  log_start "Preparing SWBD dictionary"
 
  local/swbd1_prepare_dict.sh

  log_end "Preparing SWBD dictionary"
}
#run_prep

run_prep_data(){
  # check if wav.scp text utt2spk spk2utt reco2file_and_channel exists 
  # in data/local/train . If not please copy it from 
  # /scratch2/cxy110530/SWB_KALDI_COPY/swb_kaldi/data/local/train 
  echo "Skip this step, Check the comment" 
}
#run_prep_data

run_prep_lm(){
 
  # Need to copy directory /scratch2/cxy110530/SWB_KALDI_COPY/swb_kaldi/data/lang
  # to data/lang 

  # Neet to copy directory /scratch2/cxy110530/SWB_KALDI_COPY/swb_kaldi/data/lang_test
  # to data/lang_test

  # Need to Creat directory data/eval_2000
  # and copy wav.scp text spk2utt utt2spk reco2file_and_channel from /scratch2/cxy110530/SWB_KALDI_COPY/swb_kaldi/data/eval2000_mfcc

  # Need to Creat directory data/train
  # and copy wav.scp text spk2utt utt2spk reco2file_and_channel from /scratch2/cxy110530/SWB_KALDI_COPY/swb_kaldi/data/mfcc_train

}
#run_prep_lm


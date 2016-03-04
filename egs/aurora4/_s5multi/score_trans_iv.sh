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

. cmd.sh
. path.sh
 
#set -e # exit on error
 
mkdir -p exp/trials;
mkdir -p exp/score.trans; rm -rf score.trans/*

trials=exp/trials/trial.utt2utt
trials_key=exp/trials/trial.utt2utt.keys

run_cds_score_trans-IV(){

    cat $trials | awk '{print $1, $2}' | \
    ivector-compute-dot-products - \
          scp:data/test_eval92.trans-iv/ivector.scp \
          scp:data/test_eval92.trans-iv/ivector.scp \
          exp/score.trans/cds.output 2> exp/score.trans/cds.log
    awk '{print $3}' exp/score.trans/cds.output > exp/score.trans/cds.score
    paste exp/score.trans/cds.score $trials_key > exp/score.trans/cds.score.key
    echo "CDS EER : `compute-eer exp/score.trans/cds.score.key 2> exp/score.trans/cds_EER`"
}
run_cds_score_trans-IV

run_lda_plda(){
    mkdir -p exp/score.trans/ivector_plda; rm -rf exp/score.trans/ivector_plda/*

    ivector-compute-lda --dim=50 --total-covariance-factor=0.1 \
        'ark:ivector-normalize-length scp:data/train_si84_multi.trans-iv/ivector.scp ark:- |' \
        ark:data/train_si84_multi/utt2spk \
        exp/score.trans/ivector_plda/lda_transform.mat 2> exp/score.trans/ivector_plda/lda.log

    ivector-compute-plda ark:data/train_si84_multi/spk2utt \
          'ark:ivector-transform exp/score.trans/ivector_plda/lda_transform.mat scp:data/train_si84_multi.trans-iv/ivector.scp ark:- | ivector-normalize-length ark:-  ark:- |' \
            exp/score.trans/ivector_plda/plda 2>exp/score.trans/ivector_plda/plda.log

    ivector-plda-scoring  \
           "ivector-copy-plda --smoothing=0.0 exp/score.trans/ivector_plda/plda - |" \
           "ark:ivector-transform exp/score.trans/ivector_plda/lda_transform.mat scp:data/test_eval92.trans-iv/ivector.scp ark:- | ivector-subtract-global-mean ark:- ark:- |" \
           "ark:ivector-transform exp/score.trans/ivector_plda/lda_transform.mat scp:data/test_eval92.trans-iv/ivector.scp ark:- | ivector-subtract-global-mean ark:- ark:- |" \
           "cat '$trials' | awk '{print \$1, \$2}' |" exp/score.trans/ivector_plda/plda.output 2> exp/score.trans/ivector_plda/plda.log

    awk '{print $3}' exp/score.trans/ivector_plda/plda.output > exp/score.trans/ivector_plda/plda.score
    paste exp/score.trans/ivector_plda/plda.score $trials_key > exp/score.trans/ivector_plda/plda.score.key
    echo "PLDA EER : `compute-eer exp/score.trans/ivector_plda/plda.score.key 2> exp/score.trans/ivector_plda/plda_EER`"
}
run_lda_plda


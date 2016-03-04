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
mkdir -p exp/score.ubm exp/score.tra; rm -rf score.ubm/* score.tra/*
 
make_trial(){
 
         rm -f exp/trials/trial.utt2utt; rm -f exp/trials/trial.utt2utt.keys; rm -f exp/trials/trial.utt2utt.keys;

         # Check similarity of all i-vectors belong to two desired speakers. Code below compare speaker '01jo' vs '40po'   
         for i in `cat data/test_eval92.iv/ivector.scp | awk '{print $1}' | awk 'NR == 1 || NR % 10 == 0'`;do
               for j in `cat data/test_eval92.iv/ivector.scp | awk '{print $1}' | awk 'NR == 1 || NR % 25 == 0'`;do
                   si=`grep $i data/test_eval92/utt2spk | awk '{print $2}'`
                   sj=`grep $j data/test_eval92/utt2spk | awk '{print $2}'`

                   exclude_i=`grep 'Did not' exp/tri2b_multi_ali_eval92/log/* | awk '{print $8}' | awk -F',' '{print $1}' | egrep $i `;     
                   exclude_j=`grep 'Did not' exp/tri2b_multi_ali_eval92/log/* | awk '{print $8}' | awk -F',' '{print $1}' | grep $j`; 

                   if [ -z $exclude_i ] && [ -z $exclude_j ]; then     

                           echo "$i $j" >> exp/trials/trial.utt2utt 
 
                           if [ "$si" == "$sj" ]; then
                              echo "target" >> exp/trials/trial.utt2utt.keys
                           else
                              echo "nontarget" >> exp/trials/trial.utt2utt.keys
                           fi
                  fi
               done
         done
 
}
#make_trial

trials=exp/trials/trial.utt2utt
trials_key=exp/trials/trial.utt2utt.keys

run_cds_score_UBM-IV(){

    cat $trials | awk '{print $1, $2}' | \
    ivector-compute-dot-products - \
          scp:data/test_eval92.iv/ivector.scp \
          scp:data/test_eval92.iv/ivector.scp \
          exp/score.ubm/cds.output 2> exp/score.ubm/cds.log
    awk '{print $3}' exp/score.ubm/cds.output > exp/score.ubm/cds.score
    paste exp/score.ubm/cds.score $trials_key > exp/score.ubm/cds.score.key
    echo "CDS EER : `compute-eer exp/score.ubm/cds.score.key 2> exp/score.ubm/cds_EER`"
}
run_cds_score_UBM-IV

run_lda_plda(){
    mkdir -p exp/score.ubm/ivector_plda; rm -rf exp/score.ubm/ivector_plda/*

    ivector-compute-lda --dim=50 --total-covariance-factor=0.1 \
        'ark:ivector-normalize-length scp:data/train_si84_multi.iv/ivector.scp ark:- |' \
        ark:data/train_si84_multi/utt2spk \
        exp/score.ubm/ivector_plda/lda_transform.mat 2> exp/score.ubm/ivector_plda/lda.log

    ivector-compute-plda ark:data/train_si84_multi/spk2utt \
          'ark:ivector-transform exp/score.ubm/ivector_plda/lda_transform.mat scp:data/train_si84_multi.iv/ivector.scp ark:- | ivector-normalize-length ark:-  ark:- |' \
            exp/score.ubm/ivector_plda/plda 2>exp/score.ubm/ivector_plda/plda.log

    ivector-plda-scoring  \
           "ivector-copy-plda --smoothing=0.0 exp/score.ubm/ivector_plda/plda - |" \
           "ark:ivector-transform exp/score.ubm/ivector_plda/lda_transform.mat scp:data/test_eval92.iv/ivector.scp ark:- | ivector-subtract-global-mean ark:- ark:- |" \
           "ark:ivector-transform exp/score.ubm/ivector_plda/lda_transform.mat scp:data/test_eval92.iv/ivector.scp ark:- | ivector-subtract-global-mean ark:- ark:- |" \
           "cat '$trials' | awk '{print \$1, \$2}' |" exp/score.ubm/ivector_plda/plda.output 2> exp/score.ubm/ivector_plda/plda.log

    awk '{print $3}' exp/score.ubm/ivector_plda/plda.output > exp/score.ubm/ivector_plda/plda.score
    paste exp/score.ubm/ivector_plda/plda.score $trials_key > exp/score.ubm/ivector_plda/plda.score.key
    echo "PLDA EER : `compute-eer exp/score.ubm/ivector_plda/plda.score.key 2> exp/score.ubm/ivector_plda/plda_EER`"
}
run_lda_plda

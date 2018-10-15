#!/bin/bash
#####  Train an xICFG model usign 4-block ResNets on LSUN datasets. 
###################################################
#####  NOTE: Training data is not provided.   #####
###################################################

  #----------------------------------------------------------------------------# 
  gpu=-1 # Set gpu#. -1 means "Use default"
  # Uncomment one of the 2 lines. 
  nm=brlr; imgsz=64; cc=3; batches=10 # 64x64 3-channel, 10 data batches 
#  nm=twbg; imgsz=64; cc=3; batches=7  # 64x64 3-channel,  7 data batches 

#  do_icml=1 # Uncomment this line to use the setting in [Johnson+Zhang,ICML18]
  #----------------------------------------------------------------------------#

  fnm=lsun-resnet
  param=
  mynm=$fnm   
  
  source func.sh
  gpumem=${gpu}:8  # pre-allocate 8GB device memory

  cls_fn=${clsdir}/${nm}-cls.ReNet # classifier for evaluation 
  mb=64        # mini-batch size 

  #---
  # discriminator: 4-block ResNet, batch norm
  dtyp=resnet; d_depth=4; n0d=64; d_bn=BchNorm 
  _d_net; if [ $? != 0 ]; then echo $fnm: _d_net failed.; exit 1; fi
  d_param=$_pm_ # network parameters for D

  #---
  # approximator: 4-block ResNet, batch norm
  gtyp=resnet; g_depth=$d_depth; n0g=$n0d; g_bn=$d_bn
  _g_net; if [ $? != 0 ]; then echo $fnm: _g_net failed.; exit 1; fi
  g_param=$_pm_ # network parameters for tilde{G}

  #---
  save_clk=50000  # save model files after every 100K seconds
  test_clk=5000   # test the inception score after every 5K seconds
  max_clk=100000  # stop after 100k seconds 
  gen_clk=5000; num_gen=64 # generate 64 images every 5K seconds

  cfg_T=15
  cfg_eta=1  # eta in G_t(z) <- G_{t-1}(z) + eta g_t(G_{t-1}(z))
  ss=0.00025 # step-size for updating D and tilde{G}

  param="$param cfg_eta=$cfg_eta cfg_T=$cfg_T inc=$((cfg_T*5))"
# default: cfg_U=1 cfg_N=$(( mb*10 )) cfg_x_epo=10 cfg_diff_max=40
  mynm=${mynm}-${nm}-T${cfg_T}-eta${cfg_eta}

  if [ "$do_icml" = 1 ]; then
    param="$param cfg_pool_size=$(( mb*10 )) ReUsePool"
    mynm=${mynm}-pl10Re
  fi
  
  #---
  csv_fn=${csvdir}/${mynm}.csv
  log_fn=${logdir}/${mynm}.log
  mod_fn=${moddir}/${mynm}
  gen_fn=${gendir}/${mynm}

  echo Training ... See $log_fn and $csv_fn for progress ... 
  $exe $gpumem cfg_train $param \
     trnname=${nm}-trn data_dir=$inpdir Scale-1+1 \
      num_batches=$batches datatype=imagebin x_ext=.xbin y_ext=.y  \
     _z_num_rows=100 _z_normal=1 \
     $d_param $g_param \
     _d_Rmsp _g_Rmsp _d_step_size=$ss _g_step_size=$ss \
     random_seed=1 mini_batch_size=$mb test_mini_batch_size=$mb \
     classif_fn=$cls_fn \
     save_clk=$save_clk test_clk=$test_clk max_clk=$max_clk \
     ss_clk=$ss_clk \
     num_test=10000 save_fn=$mod_fn evaluation_fn=$csv_fn \
     num_gen=$num_gen gen_clk=$gen_clk gen_fn=$gen_fn \
     > $log_fn 
  if [ $? != 0 ]; then echo $fnm: training failed.; exit 1; fi


  #---  warm-start from a saved model 
  clk=$max_clk; while [ `expr length $clk` -lt 6 ]; do clk="0$clk"; done # pad '0' to length 6
  warmfn=${mod_fn}-clk${clk}.ddg # the model saved at the end of the training above
  ss=0.0001 # step-size (as well as some of other parameters) can be changed. 
  mynm=warm$((max_clk/1000))k-${fnm}-${nm}-T${cfg_T}-eta${cfg_eta}-ss${ss}
  max_clk=25000
  save_clk=25000
  csv_fn=${csvdir}/${mynm}.csv
  log_fn=${logdir}/${mynm}.log
  mod_fn=${moddir}/${mynm}
  gen_fn=${gendir}/${mynm}

  echo Training warm-starting from a saved model ... see $log_fn and $csv_fn for progress ...
  $exe $gpumem cfg_train $param fn_for_warmstart=$warmfn \
     trnname=${nm}-trn data_dir=$inpdir Scale-1+1 \
     num_batches=$batches datatype=imagebin x_ext=.xbin y_ext=.y  \
     _d_Rmsp _g_Rmsp _d_step_size=$ss _g_step_size=$ss \
     random_seed=1 mini_batch_size=$mb test_mini_batch_size=$mb \
     classif_fn=$cls_fn \
     save_clk=$save_clk test_clk=$test_clk max_clk=$max_clk \
     ss_clk=$ss_clk \
     num_test=10000 save_fn=$mod_fn evaluation_fn=$csv_fn \
     num_gen=$num_gen gen_clk=$gen_clk gen_fn=$gen_fn \
     > $log_fn 
  if [ $? != 0 ]; then echo $fnm: training failed.; exit 1; fi


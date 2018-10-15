#!/bin/bash
#####  Train an xICFG model usign a fully-connected approximator

  #----------------------------------------------------------------------------# 
  gpu=-1 # Set gpu#. -1 means "Use default"
#  do_icml=1 # Uncomment this line to use the setting in [Johnson+Zhang,ICML18]
  #----------------------------------------------------------------------------#
  
  fnm=fc2
  param=
  mynm=$fnm    

  source func.sh

  #####
  #NOTE: Training data for brlr and twbg is not provided. 
  #####
  # Uncomment one of the 4 lines 
#  nm=brlr;  imgsz=64; cc=3; batches=10; mem=11 # 64x64 3-channel, 10 data batches
#  nm=twbg;  imgsz=64; cc=3; batches=7; mem=11  # 64x64 3-channel,  7 data batches
  nm=mnist; imgsz=32; cc=1; batches=1; mem=5  # 32x32 (0 padded from 28x28) 1-channel 
#  nm=svhn;  imgsz=32; cc=3; batches=1; mem=5  # 32x32 3-channel

  cls_fn=${clsdir}/${nm}-cls.ReNet
  mb=64        # mini-batch size 

  if [ "$nm" = "brlr" ] || [ "$nm" = "twbg" ]; then
    n0d=64         # number of D's feature maps: 64,128,256,512
    d_depth=4      # D's depth used in _d_net
    cfg_eta=0.5    # step-size in ICFG

    max_clk=100000 # stop after 100K seconds
    test_clk=5000  # test every 5K seconds 
    inc=100        # show progress every 100 updates of the approximator
  elif [ "$nm" = "svhn" ]; then
    cd $inpdir
    _fn=${nm}-trn.xbin
    _download  # download training data 
    if [ $? != 0 ]; then echo $fnm: failed to download the SVHN training data.; exit 1; fi
    cd ..

    n0d=64         # number of D's feature maps: 64,128,256
    d_depth=3
    cfg_eta=0.25

    max_clk=100000
    test_clk=5000
    inc=250
  elif [ "$nm" = "mnist" ]; then
    n0d=32         # number of D's featuer maps: 32,64,128
    d_depth=3
    cfg_eta=0.1

    max_clk=15000
    test_clk=1000
    inc=1000
  else
    echo Unknown nm: $nm; exit 1
  fi

  #---
  # discriminator: DCGAN extension with 3 conv. layers, batch norm
  dtyp=dcganx; d_bn=BchNorm 
  _d_net; if [ $? != 0 ]; then echo $fnm: _d_net failed.; exit 1; fi
  d_param=$_pm_ # network parameters for D

  #---
  # approximator: two 512-dim fully-connected layers
  gtyp=fcn; g_depth=2; n0g=512
  _g_net; if [ $? != 0 ]; then echo $fnm: _g_net failed.; exit 1; fi
  g_param=$_pm_ # network parameters for tilde{G}

  #---
  save_clk=$max_clk # save a model in the end 
#  gen_clk=$test_clk; num_gen=64 # generate 64 images when testing is done

  ss=0.0001    # learning rate for rmsprop
  cfg_T=25
  param="$param cfg_eta=$cfg_eta cfg_T=$cfg_T inc=$inc"
# default: cfg_U=1 cfg_N=$(( mb*10 )) cfg_x_epo=10 cfg_diff_max=40
  mynm=${mynm}-${nm}-T${cfg_T}

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
  $exe ${gpu}:${mem} cfg_train $param \
     trnname=${nm}-trn data_dir=$inpdir Scale-1+1 \
      num_batches=$batches datatype=imagebin x_ext=.xbin y_ext=.y  \
     _z_num_rows=100 _z_normal=1 \
     $d_param $g_param _d_Rmsp _g_Rmsp \
     _d_step_size=$ss _g_step_size=$ss \
     random_seed=1 mini_batch_size=$mb \
     classif_fn=$cls_fn \
     save_clk=$save_clk test_clk=$test_clk max_clk=$max_clk \
     num_test=10000 save_fn=$mod_fn evaluation_fn=$csv_fn \
     num_gen=$num_gen gen_clk=$gen_clk gen_fn=$gen_fn \
     > $log_fn 
  if [ $? != 0 ]; then echo $fnm: training failed.; exit 1; fi

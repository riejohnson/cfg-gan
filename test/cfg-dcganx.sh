#!/bin/bash
#####  Train an xICFG model using an extension of DCGAN
#####  (Fig.2 of [Johnson+Zhang,ICML18])

  fnm=cfg-dcganx

  gpu=-1 # gpu#.  -1 means "use default"

  source func.sh
  gpumem=${gpu}:5  # pre-allocate 5GB device memory

  # Uncomment one of the 2 lines. 
#  nm=svhn; imgsz=32; cc=3 # 32x32 3-channel
  nm=mnist; imgsz=32; cc=1 # 32x32 (0 padded from 28x28) 1-channel

  cls_fn=${clsdir}/${nm}-cls.ReNet # classifier for evaluation 
  mb=64        # mini-batch size 

  #---  svhn  
  if [ "$nm" = "svhn" ]; then
    #--  download training data. 
    cd $inpdir
    _fn=${nm}-trn.xbin
    _download
    if [ $? != 0 ]; then echo $fnm: failed to download the SVHN training data.; exit 1; fi
    cd ..

    n0d=64        # number of D's feature maps: 64,128,256
    cfg_eta=0.25  # step-size eta for ICFG 
    cfg_T=5       # T 
    test_clk=1000  # test after every 1k seconds
    max_clk=100000 # stop after 100k seconds 
    inc=250        # show progress after every 250 generator updates
  #---  mnist
  elif [ "$nm" = "mnist" ]; then
    n0d=32       # number of D's feature maps: 32,64,128
    cfg_eta=1    # step-size eta for ICFG
    cfg_T=10     # T 
    test_clk=250  # test after every 250 seconds
    max_clk=15000 # stop after 15k seconds
    inc=1000      # show progress after every 1k generator updates
  else
    echo $fnm: unknown nm: $nm; exit 1
  fi

  #---  configuration of disriminator D
  dtyp=dcganx; d_depth=3; d_bn=BchNorm # DCGAN extension with 3 blocks
                                       # and batch norm.
  _d_net; if [ $? != 0 ]; then echo $fnm: _d_net failed.; exit 1; fi
  d_param=$_pm_ # network parameters for D

  #---  configuration of approximator tilde{G}
  gtyp=$dtyp; g_depth=$d_depth; g_bn=$d_bn; n0g=$n0d
  _g_net; if [ $? != 0 ]; then echo $fnm: _g_net failed.; exit 1; fi
  g_param=$_pm_ # network parameters for tilde{G}

  #---
  save_clk=$max_clk             # save a model in the end 
  gen_clk=$test_clk; num_gen=64 # generate 64 images when testing is done

  ss=0.00025    # learning rate for rmsprop
  param="cfg_eta=$cfg_eta cfg_T=$cfg_T inc=$inc"
# default: cfg_U=1 cfg_pool_size=$(( mb*10 )) cfg_x_epo=10 cfg_diff_max=40
  mynm=${fnm}-${nm}-T${cfg_T}-eta${cfg_eta}

  #---
  csv_fn=${csvdir}/${mynm}.csv
  log_fn=${logdir}/${mynm}.log
  mod_fn=${moddir}/${mynm}
  gen_fn=${gendir}/${mynm}

  echo Training ... See $log_fn and $csv_fn for progress ... 
  $exe $gpumem cfg_train $param \
     trnname=${nm}-trn data_dir=$inpdir Scale-1+1 \
      num_batches=1 datatype=imgbin x_ext=.xbin y_ext=.y  \
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

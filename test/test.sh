#!/bin/bash

  fnm=cfg-dcganx

  gpu=-1 # gpu#.  -1 means "use default"

  source func.sh
  gpumem=${gpu}:4  # pre-allocate 4GB device memory

  imgsz=32; cc=1

  #---  configuration of discriminator D
  dtyp=dcganx; d_depth=3; n0d=32; d_bn=BchNorm
  _d_net; if [ $? != 0 ]; then echo $fnm: _d_net failed.; exit 1; fi
  d_param=$_pm_ # network parameters for D

  #---  configuration of approximator tilde{G}
  gtyp=dcganx; g_depth=3; n0g=32; g_bn=BchNorm
  _g_net; if [ $? != 0 ]; then echo $fnm: _g_net failed.; exit 1; fi
  g_param=$_pm_ # network parameters for tilde{G}

  #---  training 
  $exe $gpumem cfg_train \
     trnname=${inpdir}/mnist-trn Scale-1+1 \
     datatype=imgbin x_ext=.xbin y_ext=.y  \
     cfg_T=10 cfg_eta=1 \
     _z_num_rows=100 _z_normal=1 \
     $d_param $g_param _d_Rmsp _g_Rmsp \
     _d_step_size=0.00025 _g_step_size=0.00025 \
     mini_batch_size=64 \
     classif_fn=${clsdir}/mnist-cls.ReNet \
     test_clk=10 max_clk=10 inc=10 \
     num_test=10000 
  if [ $? != 0 ]; then echo $fnm: training failed.; exit 1; fi

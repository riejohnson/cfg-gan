
  exe=../bin/cfgexe
  prepImg=../bin/prepImg

  inpdir=data
  clsdir=for_eval

  logdir=log
  moddir=mod
  csvdir=csv
  gendir=gen

  if [ ! -e $logdir ]; then mkdir $logdir; fi
  if [ ! -e $moddir ]; then mkdir $moddir; fi
  if [ ! -e $csvdir ]; then mkdir $csvdir; fi
  if [ ! -e $gendir ]; then mkdir $gendir; fi

#----------------------------------------------
# input: _fn
_download() {
  local fnm=_download

  if [ "$_fn" = "" ]; then echo _fn is missing.; return 1; fi
  if [ ! -e $_fn ]; then 
    wget http://riejohnson.com/software/cfg/${_fn}.gz
    if [ $? != 0 ]; then echo $fnm: wget failed. Check internet connection.; exit 1; fi
    echo Decompressing $_fn
    gzip -d -f ${_fn}.gz
    if [ $? != 0 ]; then echo $fnm: gzip -d failed.; return 1; fi
  fi
  return 0
}

#----------------------------------------------
_d_net() { 
  local fnm=_d_net

  if   [ "$dtyp" = "dcganx" ]; then _d_dcganx
  elif [ "$dtyp" = "resnet" ]; then _d_resnet
  else echo $fnm: unknown dtyp.  use dcganx or resnet.; return 1
  fi
  if [ $? != 0 ]; then echo $fnm: failed.; return 1; fi

  _pm_="$_pm_ _d_DivEach _d_init_weight=0.01"
  return 0
}

_g_net() { 
  local fnm=_g_net

  if   [ "$gtyp" = "dcganx" ]; then _g_dcganx
  elif [ "$gtyp" = "resnet" ]; then _g_resnet
  elif [ "$gtyp" = "fcn"    ]; then _g_fcn
  else echo $fnm: unknown gtyp.  use dcganx, resnet, or fcn.; return 1
  fi
  if [ $? != 0 ]; then echo $fnm: failed.; return 1; fi

  _pm_="$_pm_ _g_DivEach _g_init_weight=0.01"
  return 0
}


#----------------------------------------------
# output: _pm_
_d_resnet () {
  local fnm=_d_resnet

  if [ "$n0d"     = "" ]; then echo $fnm: n0d is missing; return 1; fi
  if [ "$imgsz"   = "" ]; then echo $fnm: imgsz is missing; return 1; fi
  if [ "$d_depth" = "" ]; then echo $fnm: d_depth is missing; return 1; fi
  if [ "$d_bn"    = "" ]; then echo $fnm: d_bn is missing; return 1; fi
  local bn=$d_bn

  local nn=$n0d
  local sz=$imgsz

  #-- dws[i]=1: do downsampling in block#i
  #--        0: no downsampling in block#i
  #-- d_muls[i]=2: double the number of feature maps in block#i
  #--           1: don't change the number of feature maps in block#i. 
  if [ "$d_depth" = 4 ]; then 
    local dws=(    1 1 1 1 )
    local d_muls=( 1 2 2 2 ) 
  elif [ "$d_depth" = 5 ]; then 
    local dws=(    1 1 0 1 1 )
    local d_muls=( 1 2 1 2 2 ) 
  elif [ "$d_depth" = 6 ]; then 
    local dws=(    1 1 0 1 0 1 )
    local d_muls=( 1 2 1 2 1 2 )
  elif [ "$d_depth" = 7 ]; then 
    local dws=(    1 0 1 0 1 0 1 )
    local d_muls=( 1 1 2 1 2 1 2 ) 
  elif [ "$d_depth" = 3 ]; then
    local dws=(    1 1 1 )
    local d_muls=( 1 2 2 ) 
  else
    echo $fnm: d_depth must be between 3 and 7.; return 1
  fi

  local pchsz=3
  local pad=$(( (pchsz-1)/2 ))  # padding for stride 1

  local p
  p="_d_TopThru _d_activ_type=None"
  p="$p _d_patch_size=$pchsz  _d_patch_stride=1 _d_padding=$pad"
  p="$p _d_pooling_type=Avg _d_pooling_size=2 _d_pooling_stride=2"  
  local x
  local _u=4
  local d=0 
  for (( d=0; d<d_depth; ++d )); do
    local dw=${dws[d]}     
    if [ "$dw" != 1 ] && [ "$dw" != 0 ]; then echo $fnm: dws must be 1 or 0; return 1; fi
  done
  for (( d=0; d<d_depth;  ++d )); do
    local mul=${d_muls[d]}
    if [ "$mul" != 2 ] && [ "$mul" != 1 ]; then echo $fnm: d_muls must be 1 or 2; return 1; fi
  done
  
  d=0
  local c="_d_conn=0" # main connection
  local cs=""         # shortcut 

  local l=0; x=_d_$l; p="$p ${x}layer_type=Noop ${x}name=NoopI"

  for (( d=0; d<d_depth; ++d )); do
    cs="$cs,$l"  
    local dw=${dws[d]}  
    local mul=${d_muls[d]}
    #- a
    if [ "$d" != 0 ]; then
      l=$((l+1));x=_d_$l;c=$c-$l;p="$p ${x}layer_type=Act   ${x}name=aAct$d ${x}activ_type=Rect"
    fi
    l=$((l+1));x=_d_$l;c=$c-$l;p="$p ${x}layer_type=Patch2D ${x}name=aPch$d ${x}size_x=$sz"   
    l=$((l+1));x=_d_$l;c=$c-$l;p="$p ${x}layer_type=DenWei  ${x}name=aWei$d ${x}nodes=$nn"
    #---

    nn=$((nn*mul))

    #- b
    l=$((l+1));x=_d_$l;c=$c-$l;p="$p ${x}layer_type=Act     ${x}name=bAct$d ${x}activ_type=Rect"
    l=$((l+1));x=_d_$l;c=$c-$l;p="$p ${x}layer_type=Patch2D ${x}name=bPch$d ${x}size_x=$sz"   
    l=$((l+1));x=_d_$l;c=$c-$l;p="$p ${x}layer_type=DenWei  ${x}name=bWei$d ${x}nodes=$nn"
    #---

    l=$((l+1));x=_d_$l;c=$c-$l;p="$p ${x}layer_type=${bn}   ${x}name=b${bn}$d"    
    
    if [ "$dw" = 1 ]; then
      l=$((l+1));x=_d_$l;c=$c-$l;p="$p ${x}layer_type=Pooling2D   ${x}name=bPool$d ${x}size_x=$sz"
      #-- shortcut  
      l=$((l+1));x=_d_$l;cs=$cs-$l;p="$p ${x}layer_type=Pooling2D ${x}name=sPool$d ${x}size_x=$sz"              
      l=$((l+1));x=_d_$l;cs=$cs-$l;p="$p ${x}layer_type=DenWei    ${x}name=sWei$d ${x}nodes=$nn"  
      sz=$((sz/2))
    elif [ "$mul" != 1 ]; then
      #-- shortcut
      l=$((l+1));x=_d_$l;cs=$cs-$l;p="$p ${x}layer_type=DenWei   ${x}name=sWei$d ${x}nodes=$nn"        
    fi     
    cs=$cs-$((l+1)) 
    l=$((l+1));x=_d_$l;c=$c-$l;p="$p ${x}layer_type=Noop ${x}name=Noop$d"      
  done

  l=$((l+1));x=_d_$l;c=$c-$l;p="$p ${x}layer_type=Act     ${x}name=ActAfter ${x}activ_type=Rect"
  l=$((l+1));x=_d_$l;c=$c-$l;p="$p ${x}layer_type=Patch2D ${x}name=Flatten  ${x}patch_size=$sz ${x}patch_stride=1 ${x}padding=0 ${x}size_x=$sz"
  l=$((l+1));x=_d_$l;c=$c-$l;p="$p ${x}layer_type=DenWei  ${x}name=LastWei  ${x}nodes=1"

  c="$c-top"
  p="$p $c$cs"
  p="$p _d_layers=$((l+1)) _d_AdditiveConn"

  _pm_=$p  # return values by the global variable _pm_
  
  return 0
}


#---------------------------------------------- 
# output: _pm_
_g_resnet () {
  local fnm=_g_resnet

  if [ "$n0g"   = "" ]; then echo $fnm: n0g is missing; return 1; fi
  if [ "$imgsz" = "" ]; then echo $fnm: imgsz is missing; return 1; fi
  if [ "$g_depth" = "" ]; then echo $fnm: g_depth is missing; return 1; fi
  if [ "$cc"      = "" ]; then echo $fnm: cc is missing; return 1; fi
  if [ "$g_bn"    = "" ]; then echo $fnm: g_bn is missing; return 1; fi
  local bn=$g_bn

  #-- ups[i]=1: do upsampling in block#i
  #--        0: no upsampling in block#i
  #-- g_muls[i]=2: halve the number of feature maps in block#i
  #--           1: don't change the number of feature maps in block#i. 
  if [ "$g_depth" = 4 ]; then 
    local ups=(    1 1 1 1 )
    local g_muls=( 2 2 2 1 )
  elif [ "$g_depth" = 5 ]; then 
    local ups=(    1 1 0 1 1 )
    local g_muls=( 2 2 1 2 1 )
  elif [ "$g_depth" = 6 ]; then 
    local ups=(    1 0 1 0 1 1 )
    local g_muls=( 2 1 2 1 2 1 )
  elif [ "$g_depth" = 7 ]; then 
    local ups=(    1 0 1 0 1 0 1 )
    local g_muls=( 2 1 2 1 2 1 1 )
  elif [ "$g_depth" = 3 ]; then
    local ups=(    1 1 1 )
    local g_muls=( 2 2 1 )
  else
    echo $fnm: g_depth must be between 3 and 7.; return 1
  fi

  local pchsz=3
  local pad=$(( (pchsz-1)/2 ))  # padding for stride 1

  local i; 
  local nn=$n0g
  local sz=$imgsz; 
  for (( i=0; i<g_depth;   ++i )); do 
    local up=${ups[i]}     
    if [ "$up" != 1 ] && [ "$up" != 0 ]; then echo $fnm: ups must be 1 or 0; return 1; fi  
    if [ "$up" = 1 ]; then sz=$((sz/2)); fi
  done 
  for (( i=0; i<g_depth;  ++i )); do
    local mul=${g_muls[i]}
    if [ "$mul" != 2 ] && [ "$mul" != 1 ]; then echo $fnm: g_muls must be 1 or 2; return 1; fi
    nn=$((nn*mul))
  done

  local p
  p="_g_TopThru _g_activ_type=None _g_patch_size=$pchsz _g_patch_stride=1 _g_padding=$pad"
  p="$p _g_pooling_type=Max _g_pooling_size=2 _g_pooling_stride=2"

  local l; local x
  local c="_g_conn=0"
  local cs=""
  l=0;       x=_g_$l;        p="$p ${x}layer_type=Noop    ${x}name=Noop"
  l=$((l+1));x=_g_$l;c=$c-$l;p="$p ${x}layer_type=DenWei  ${x}name=Project ${x}nodes=$((nn*sz*sz))"
  l=$((l+1));x=_g_$l;c=$c-$l;p="$p ${x}layer_type=Reshape ${x}name=Reshape ${x}num_rows=$nn"
  local d; for (( d=0; d<g_depth; ++d )); do
    cs="$cs,$l"
    local up=${ups[d]}   
    local mul=${g_muls[d]}

    #- a
    l=$((l+1));x=_g_$l;c=$c-$l;p="$p ${x}layer_type=Act ${x}name=aAct$d ${x}activ_type=Rect"
    if [ "$up" = 1 ]; then 
      sz=$((sz*2))
      l=$((l+1));x=_g_$l;c=$c-$l;p="$p ${x}layer_type=Pooling2D ${x}name=aPlup$d ${x}Transpose ${x}size_x=$sz"    
    fi
    l=$((l+1));x=_g_$l;c=$c-$l;p="$p ${x}layer_type=Patch2D ${x}name=aPch$d ${x}size_x=${sz}"    
    nn=$((nn/mul))
    l=$((l+1));x=_g_$l;c=$c-$l;p="$p ${x}layer_type=DenWei  ${x}name=aWei$d ${x}nodes=$nn"

    #- b
    l=$((l+1));x=_g_$l;c=$c-$l;p="$p ${x}layer_type=Act     ${x}name=bAct$d ${x}activ_type=Rect"
    l=$((l+1));x=_g_$l;c=$c-$l;p="$p ${x}layer_type=Patch2D ${x}name=bPch$d ${x}size_x=${sz}"    
    l=$((l+1));x=_g_$l;c=$c-$l;p="$p ${x}layer_type=DenWei  ${x}name=bWei$d ${x}nodes=$nn"    

    l=$((l+1));x=_g_$l;c=$c-$l;p="$p ${x}layer_type=$bn     ${x}name=b$bn$d" 
    
    if [ "$up" = 1 ]; then
      l=$((l+1));x=_g_$l;cs=$cs-$l;p="$p ${x}layer_type=DenWei    ${x}name=sWei$d  ${x}nodes=$nn"        
      l=$((l+1));x=_g_$l;cs=$cs-$l;p="$p ${x}layer_type=Pooling2D ${x}name=sPlup$d ${x}Transpose ${x}size_x=$sz"  
    elif [ "$mul" != 1 ]; then
      l=$((l+1));x=_g_$l;cs=$cs-$l;p="$p ${x}layer_type=DenWei    ${x}name=sWei$d  ${x}nodes=$nn"      
    fi
    cs=$cs-$((l+1))
    l=$((l+1));x=_g_$l;c=$c-$l;p="$p ${x}layer_type=Noop  ${x}name=Noop$d"   
  done  
  l=$((l+1));x=_g_$l;c=$c-$l;p="$p ${x}layer_type=Act     ${x}name=LastAct ${x}activ_type=Rect"
  l=$((l+1));x=_g_$l;c=$c-$l;p="$p ${x}layer_type=Patch2D ${x}name=LastPch ${x}size_x=$sz"
  l=$((l+1));x=_g_$l;c=$c-$l;p="$p ${x}layer_type=DenWei  ${x}name=LastWei ${x}nodes=$cc"
  if [ "$sz" != "$imgsz" ]; then echo $fnm: sz=$sz imgsz=$imgsz   no match; return 1; fi
  l=$((l+1));x=_g_$l;c=$c-$l;p="$p ${x}layer_type=Act ${x}name=Tanh ${x}activ_type=Tanh"

  c="$c-top"
  p="$p $c$cs"
  p="$p _g_layers=$((l+1)) _g_AdditiveConn"

  _pm_=$p  # return values by the global variable _pm_

  return 0
}


#----------------------------------------------
# output: _pm_: DCGAN extension
_d_dcganx () {
  local fnm=_d_dcganx

  if [ "$n0d"   = "" ]; then echo $fnm: n0d is missing; return 1; fi
  if [ "$imgsz" = "" ]; then echo $fnm: imgsz is missing; return 1; fi
  if [ "$d_depth" = "" ]; then echo $fnm: d_depth is missing; return 1; fi
  if [ "$d_bn"    = "" ]; then echo $fnm: d_bn is missing; return 1; fi
  local bn=$d_bn

  local d_extra=1
  if [ "$d_nodemul"   = "" ]; then d_nodemul=2; fi

  local psz=5; local pad=1
  local _nn=$n0d
  local _sz=$imgsz

  local p
  p="_d_TopThru"; 
  p="$p _d_activ_slope=0.2 _d_patch_size=$psz _d_patch_stride=2 _d_padding=$pad"
  local l; local x
  l=0;       x=_d_$l;p="$p ${x}layer_type=Noop ${x}name=Noop"
  l=$((l+1));x=_d_$l;p="$p ${x}layer_type=Patch2D ${x}name=Patch$d  ${x}size_x=$_sz"
  l=$((l+1));x=_d_$l;p="$p ${x}layer_type=DenWei  ${x}name=Weight$d ${x}nodes=$_nn"
  l=$((l+1));x=_d_$l;p="$p ${x}layer_type=Act     ${x}name=Act$d    ${x}activ_type=Rect"
  _nn=$((_nn*d_nodemul)); 
  _sz=$((_sz/2))  

  for (( d=1; d<d_depth; ++d )); do
    l=$((l+1));x=_d_$l;p="$p ${x}layer_type=Patch2D ${x}name=Patch$d  ${x}size_x=$_sz"
    l=$((l+1));x=_d_$l;p="$p ${x}layer_type=DenWei  ${x}name=Weight$d ${x}nodes=$_nn"
    l=$((l+1));x=_d_$l;p="$p ${x}layer_type=$bn     ${x}name=$bn$d"
    l=$((l+1));x=_d_$l;p="$p ${x}layer_type=Act     ${x}name=Act$d    ${x}activ_type=Rect"
    if [ "$d_extra" != "" ]; then
      local i; for (( i=0; i<d_extra; ++i )); do
        l=$((l+1));x=_d_$l;p="$p ${x}layer_type=DenWei ${x}name=WeiX$d   ${x}nodes=$_nn"
        l=$((l+1));x=_d_$l;p="$p ${x}layer_type=$bn    ${x}name=${bn}X$d"
        l=$((l+1));x=_d_$l;p="$p ${x}layer_type=Act    ${x}name=ActX$d   ${x}activ_type=Rect"       
      done
    fi
    _nn=$((_nn*d_nodemul)); 
    _sz=$((_sz/2))
  done

  l=$((l+1));x=_d_$l;p="$p ${x}layer_type=Patch2D ${x}name=Flatten ${x}patch_size=$_sz ${x}patch_stride=1 ${x}padding=0 ${x}size_x=$_sz"
  if [ "$doing_vae" != 1 ]; then
    l=$((l+1));x=_d_$l;p="$p ${x}layer_type=DenWei  ${x}name=LastWeight ${x}nodes=1"
  fi

  local c="_d_conn=0"
  for (( i=1; i<=l; ++i )); do c="$c-$i"; done
  c="$c-top"

  p="$p $c _d_layers=$((l+1))"

  _pm_=$p  # return values by the global variable _pm_

  return 0
}

#----------------------------------------------
# output: _pm_
_g_dcganx () {
  local fnm=_g_dcganx

  local g_extra=1

  if [ "$cc"    = "" ]; then echo $fnm: cc is missing; return 1; fi    # number of channels
  if [ "$n0g"   = "" ]; then echo $fnm: n0g is missing; return 1; fi   # number of feature maps 
  if [ "$imgsz" = "" ]; then echo $fnm: imgsz is missing; return 1; fi # image size 
  if [ "$g_depth" = "" ]; then echo $fnm: g_depth is missing; return 1; fi # depth 
  if [ "$g_bn"    = "" ]; then echo $fnm: g_bn is missing; return 1; fi # BNorm or Noop
  local bn=$g_bn

  if [ "$g_nodemul"   = "" ]; then g_nodemul=2; fi

  local i; 
  local _nn=$n0g;   for (( i=0; i<g_depth-1; ++i )); do _nn=$((_nn*g_nodemul)); done
  local _sz=$imgsz; for (( i=0; i<g_depth;   ++i )); do _sz=$((_sz/2)); done

  local psz=5

  local p; 
  p="_g_TopThru _g_activ_type=None _g_patch_size=$psz _g_patch_stride=2 _g_padding=1"

  local l; local x
  l=0;       x=_g_$l;p="$p ${x}layer_type=Noop    ${x}name=Noop"
  l=$((l+1));x=_g_$l;p="$p ${x}layer_type=DenWei  ${x}name=Project   ${x}nodes=$((_nn*_sz*_sz))"
  l=$((l+1));x=_g_$l;p="$p ${x}layer_type=Reshape ${x}name=Reshape   ${x}num_rows=$_nn"
  _nn=$((_nn/g_nodemul)); _sz=$((_sz*2))

  local d; for (( d=0; d<g_depth-1; ++d )); do
    l=$((l+1));x=_g_$l;p="$p ${x}layer_type=Act     ${x}name=Act$d      ${x}activ_type=Rect"
    l=$((l+1));x=_g_$l;p="$p ${x}layer_type=DenWei  ${x}name=Weight$d   ${x}nodes=$((_nn*psz*psz))"
    l=$((l+1));x=_g_$l;p="$p ${x}layer_type=Patch2D ${x}name=Patch2Dt$d ${x}Transpose ${x}size_x=$_sz"
    l=$((l+1));x=_g_$l;p="$p ${x}layer_type=$bn     ${x}name=$bn$d"

    if [ "$g_extra" != "" ]; then
      for (( i=0; i<g_extra; ++i )); do
        l=$((l+1));x=_g_$l;p="$p ${x}layer_type=Act    ${x}name=ActX$d   ${x}activ_type=Rect"    
        l=$((l+1));x=_g_$l;p="$p ${x}layer_type=DenWei ${x}name=WeiX$d   ${x}nodes=$_nn"
        l=$((l+1));x=_g_$l;p="$p ${x}layer_type=${bn}  ${x}name=${bn}X$d"  
      done
    fi

    _nn=$((_nn/g_nodemul)); _sz=$((_sz*2))
  done

  l=$((l+1));x=_g_$l;p="$p ${x}layer_type=Act     ${x}name=LastAct    ${x}activ_type=Rect"
  l=$((l+1));x=_g_$l;p="$p ${x}layer_type=DenWei  ${x}name=LastWeight ${x}nodes=$((cc*psz*psz))"
  l=$((l+1));x=_g_$l;p="$p ${x}layer_type=Patch2D ${x}name=LastPatch  ${x}Transpose ${x}size_x=$imgsz"
  l=$((l+1));x=_g_$l;p="$p ${x}layer_type=Act     ${x}name=Tanh       ${x}activ_type=Tanh"

  local c="_g_conn=0"
  for (( i=1; i<=l; ++i )); do c="$c-$i"; done
  c="$c-top"

  p="$p $c _g_layers=$((l+1))"

  _pm_=$p  # return values by the global variable _pm_

  return 0
}

#----------------------------------------------
# output: _pm_
_g_fcn () {
  local fnm=_g_fcn

  if [ "$n0g"   = "" ]; then echo $fnm: n0g is missing; return 1; fi
  if [ "$imgsz" = "" ]; then echo $fnm: imgsz is missing; return 1; fi
  if [ "$g_depth" = "" ]; then echo $fnm: g_depth is missing; return 1; fi
  if [ "$cc"    = "" ]; then echo $fnm: cc is missing; return 1; fi

  local p
  p="_g_TopThru"

  local l=-1
  local d 
  for (( d=0; d<g_depth; ++d )); do
    l=$((l+1));x=_g_$l;p="$p ${x}layer_type=DenWei ${x}name=Wei$d ${x}nodes=$n0g"
    l=$((l+1));x=_g_$l;p="$p ${x}layer_type=Act    ${x}name=Act$d ${x}activ_type=Rect"
  done
  l=$((l+1));x=_g_$l;p="$p ${x}layer_type=Weight  ${x}name=LastWeight ${x}nodes=$((cc*imgsz*imgsz))"
  l=$((l+1));x=_g_$l;p="$p ${x}layer_type=Act     ${x}name=Tanh       ${x}activ_type=Tanh"
  l=$((l+1));x=_g_$l;p="$p ${x}layer_type=Reshape ${x}name=Reshape    ${x}num_rows=$cc"

  p="$p _g_layers=$((l+1))"

  _pm_=$p  # return values by the global variable _pm_

  return 0
}



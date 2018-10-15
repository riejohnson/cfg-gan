####  Change CUDA_PATH and -gencode options in CFLAGS1 if necessary. 
####
####  This makefile works fine with CUDA 8.0 and Maxwell/Pascal GPUs. 
####
####  If your CUDA version is older, you may need to remove some "-gencode" 
####  line(s) that your CUDA does not support, to avoid compile error. 
####
####  If the compute capability of your GPU card is higher than 6.0, 
####  you need to add a "-gencode" line for it. 
####
####  The compute capability of your GPU can be found by looking up Wikipedia:
####       https://en.wikipedia.org/wiki/CUDA . 
####  It can be also found by entering "gpuDevice" in matlab.  
####

SHELL = /bin/bash

BIN_NAME1 = cfgexe
BIN_DIR = .
TARGET1 = $(BIN_DIR)/$(BIN_NAME1)

#<=== Change CUDA_PATH
CUDA_PATH=/usr/local/cuda
#CUDA_PATH = /opt/sw/packages/cuda/8.0

CUDA_INC_PATH   = $(CUDA_PATH)/include
CUDA_BIN_PATH   = $(CUDA_PATH)/bin
CUDA_LIB_PATH   = $(CUDA_PATH)/lib64
LDFLAGS1   = -L$(CUDA_LIB_PATH) -lcudart -lcublas -lcurand -lcusparse

#<=== add or remove -gencode as needed. 
CFLAGS1 = -Isrc/com -Isrc/data -Isrc/nnet  -D__AZ_SMAT_SINGLE__ -D__AZ_GPU__  -I$(CUDA_INC_PATH) -O2 \
		-gencode arch=compute_30,code=sm_30 \
		-gencode arch=compute_32,code=sm_32 \
		-gencode arch=compute_35,code=sm_35 \
		-gencode arch=compute_37,code=sm_37 \
		-gencode arch=compute_50,code=sm_50 \
		-gencode arch=compute_52,code=sm_52 \
		-gencode arch=compute_53,code=sm_53 \
		-gencode arch=compute_60,code=sm_60 \
		-gencode arch=compute_61,code=sm_61 \
		-gencode arch=compute_62,code=sm_62

CPP_FILES1= 	\
	src/img/main_cfg.cpp \
	src/img/AzpG_ClasEval.cpp \
	src/img/AzpMain_Cfg.cpp \
	src/img/AzpG_Tools.cpp \
	src/img/AzpG_Cfg.cpp \
	src/nnet/AzpPatchDflt.cpp \
	src/nnet/AzpReNet.cpp \
	src/nnet/AzpReLayer.cpp \
	src/nnet/AzMultiConn.cpp \
	src/nnet/AzpMain_reNet.cpp \
	src/nnet/AzpLmSgd.cpp \
	src/nnet/AzPmat.cpp \
	src/nnet/AzPmatSpa.cpp \
	src/nnet/AzPmatApp.cpp \
	src/nnet/AzPmat_gpu.cu \
	src/nnet/AzPmatSpa_gpu.cu \
	src/nnet/AzCuda_Pmat.cu \
	src/nnet/AzCuda_PmatSpa.cu \
	src/nnet/AzCuda_PmatApp.cu \
	src/nnet/AzpEv.cpp \
	src/nnet/AzpLossDflt.cpp \
	src/com/AzUtil.cpp \
	src/com/AzDmat.cpp \
	src/com/AzParam.cpp \
	src/com/AzSmat.cpp \
	src/com/AzStrPool.cpp \
	src/com/AzTextMat.cpp \
	src/com/AzTools.cpp

BIN_NAME2 = prepImg
TARGET2 = $(BIN_DIR)/$(BIN_NAME2)
CFLAGS2 = -Isrc/com -O2 -D__AZ_SMAT_SINGLE__

TARGET3 = ../test/data/mnist-trn.xbin

CPP_FILES2= 	\
	src/com/AzUtil.cpp \
	src/img/main_prepImg.cpp

all: $(TARGET1) $(TARGET2) $(TARGET3)
$(TARGET1): 
	rm -f $(TARGET1)
	$(CUDA_BIN_PATH)/nvcc $(CPP_FILES1) $(CFLAGS1) -o $(TARGET1) $(LDFLAGS1)

$(TARGET2):
	rm -f $(TARGET2)
	g++ $(CPP_FILES2) $(CFLAGS2) -o $(TARGET2)

$(TARGET3):
	chmod +x ../test/*.sh
	if [ ! -e $(TARGET3) ]; then gzip -d $(TARGET3).gz; fi

clean: 
	rm -f $(TARGET1)
	rm -f $(TARGET2)

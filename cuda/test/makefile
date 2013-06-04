# Adapted from the CUDA 5 Toolkit samples.

# OS Name (Linux or Darwin)
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OSLOWER = $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])

# Flags to detect 32-bit or 64-bit OS platform
OS_SIZE = $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/")
OS_ARCH = $(shell uname -m | sed -e "s/i386/i686/")

dbg=1

# These flags will override any settings
ifeq ($(i386),1)
	OS_SIZE = 32
	OS_ARCH = i686
endif

ifeq ($(x86_64),1)
	OS_SIZE = 64
	OS_ARCH = x86_64
endif

# Flags to detect either a Linux system (linux) or Mac OSX (darwin)
DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))

# Location of the CUDA Toolkit binaries and libraries
CUDA_PATH       ?= /usr/local/cuda-5.0
CUDA_INC_PATH   ?= -I$(CUDA_PATH)/include -I/home/sarma/lib/inc #Samples\v5.0\C\common\inc
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
ifneq ($(DARWIN),)
  CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib
else
  ifeq ($(OS_SIZE),32)
    CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib
  else
    CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib64
  endif
endif

# Common binaries
NVCC            ?= $(CUDA_BIN_PATH)/nvcc
GCC             ?= g++

# Extra user flags
EXTRA_NVCCFLAGS ?=
EXTRA_LDFLAGS   ?=
EXTRA_CCFLAGS   ?=

# CUDA code generation flags
GENCODE_SM10    := -gencode arch=compute_10,code=sm_10
GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35
GENCODE_FLAGS   := $(GENCODE_SM10) $(GENCODE_SM20) $(GENCODE_SM30)

# OS-specific build flags
ifneq ($(DARWIN),) 
      LDFLAGS   := -Xlinker -rpath $(CUDA_LIB_PATH) -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -arch $(OS_ARCH) 
else
  ifeq ($(OS_SIZE),32)
      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -m32
  else
      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -m64
  endif
endif

# OS-architecture specific flagsRCH
#
ifeq ($(OS_SIZE),32)
      NVCCFLAGS := -m32
else
      NVCCFLAGS := -m64
endif

# Debug build flags
ifeq ($(dbg),1)
      CCFLAGS   += -g
      NVCCFLAGS += -g -G
      TARGET    := debug
else
      TARGET    := release
endif


# Common includes and paths for CUDA
INCLUDES      := -I$(CUDA_INC_PATH) -I. -I.. -I../../common/inc
LDFLAGS       += -lcudart -lcufft

# Conv Include
CONVINC				:= -I../include
CONVLIB				:= ../lib/cudaConv.o
CUFFT					:= -lcufft

# Target rules
all: build

build: test.out
#build: cudaConv.out
#build: cudaConv.o

test.o: test.cu
	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(CONVINC) $(CONVLIB) $(CUFFT) $(INCLUDES) $(GENCODE_FLAGS) -o $@ $<

test.out: test.o
	$(GCC) $(CCFLAGS) -o $@ $+ $(CONVLIB) $(CUFFT) $(LDFLAGS) $(EXTRA_LDFLAGS)
	mkdir -p ../../bin/$(OSLOWER)/$(TARGET)
	cp $@ ../../bin/$(OSLOWER)/$(TARGET)

run: main
	./test.out

clean:
	rm -f test.o

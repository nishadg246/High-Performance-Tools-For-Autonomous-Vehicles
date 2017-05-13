
EXECUTABLE := slam

CU_FILES   := cudaSLAM.cu

CU_DEPS    :=

CC_FILES   := main.cpp noise.cpp

LOGS	   := logs

###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')
OBJDIR=objs
CXX=g++ -m64 -I ~/slam/eigen 
CXXFLAGS=-O3 -Wall -g
HOSTNAME=$(shell hostname)

LIBS       :=
FRAMEWORKS :=

NVCCFLAGS=-O3 -m64 --gpu-architecture compute_35 -I ~/slam/eigen 
LIBS += GL glut cudart

ifneq ($(wildcard /opt/cuda-7.5/.*),)
# Latedays
LDFLAGS=-L/opt/cuda-7.5/lib64/ -lcudart
else
# GHC
LDFLAGS=-L/usr/local/cuda/lib64/ -lcudart
endif

LDLIBS  := $(addprefix -l, $(LIBS))
LDFRAMEWORKS := $(addprefix -framework , $(FRAMEWORKS))

NVCC=nvcc

OBJS=$(OBJDIR)/main.o $(OBJDIR)/noise.o $(OBJDIR)/cudaSLAM.o 


.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *~ $(EXECUTABLE) $(LOGS)

check:	default
		./checker.pl

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS) $(LDFRAMEWORKS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@

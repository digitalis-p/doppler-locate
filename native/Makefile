# Builds src/dopploc/_native/libdopploc_native.so
#   make        CUDA + OpenMP (needs nvcc)
#   make cpu    OpenMP only
CUDA ?= /opt/cuda
NVCC ?= $(CUDA)/bin/nvcc
CXX ?= g++
OUT := ../src/dopploc/_native/libdopploc_native.so
HOSTFLAGS := -O3 -march=native -fopenmp -fPIC

all: $(OUT)

$(OUT): dopploc_native.cu dopploc_search.cu dopploc_native.h
	mkdir -p $(dir $(OUT))
	$(NVCC) -O3 -std=c++17 -shared -arch=native --use_fast_math -allow-unsupported-compiler \
	    -Xcompiler "$(HOSTFLAGS)" -o $@ dopploc_native.cu dopploc_search.cu -lgomp

cpu: dopploc_native.cu dopploc_native.h
	mkdir -p $(dir $(OUT))
	$(CXX) $(HOSTFLAGS) -std=c++17 -shared -x c++ -o $(OUT) dopploc_native.cu

clean:
	rm -f $(OUT)

.PHONY: all cpu clean

NVCC = nvcc
NVCC_FLAGS = -arch=sm_70 -O2 -std=c++17

all: ghost_tiles_cuda.o

ghost_tiles_cuda.o: src/ghost_tiles.cu include/ghost_tiles.cuh
	mkdir -p build
	$(NVCC) $(NVCC_FLAGS) -c src/ghost_tiles.cu -o build/ghost_tiles_cuda.o

clean:
	rm -rf build

.PHONY: all clean

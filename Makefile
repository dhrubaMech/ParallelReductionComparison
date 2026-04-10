
NVCC      = nvcc
NVCCFLAGS = -O3 -Xcompiler -fopenmp  -Wno-deprecated-gpu-targets -lineinfo 
LDFLAGS   = -lcudart

CXX      = g++
CXXFLAGS = -fopenmp

TARGET  = res
SOURCES = main.cu kernels.cu helperFunctions.cpp
OBJECTS = main.o  kernels.o  helperFunctions.o

$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS) 


main.o: main.cu
	$(NVCC) $(NVCCFLAGS) -c main.cu -o main.o

kernels.o: kernels.cu
	$(NVCC) $(NVCCFLAGS) -c kernels.cu -o kernels.o

helperFunctions.o: helperFunctions.cpp
	$(NVCC) $(NVCCFLAGS) -c helperFunctions.cpp -o helperFunctions.o

clean:
	rm -f $(OBJECTS) $(TARGET)


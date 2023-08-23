BIN_DIR = build
GPU_ARCH = compute_75
COMPUTE_CODE = sm_75

default:
	nvcc my_gemm.cu -o $(BIN_DIR)/gemm -arch=$(GPU_ARCH) -code=$(COMPUTE_CODE)

gemm:
	nvcc my_gemm.cu -o $(BIN_DIR)/gemm -arch=$(GPU_ARCH) -code=$(COMPUTE_CODE)
	nvcc cublas_sgemm.cu -o $(BIN_DIR)/cublas_test -arch=$(GPU_ARCH) -code=$(COMPUTE_CODE) -lcublas

a100:
	nvcc cublas_sgemm.cu -o $(BIN_DIR)/cublas_test -arch=compute_80 -code=sm_80 -lcublas
	
clean:
	rm -rf build/

#include"MyMacro.h"
#include"MyProto.h"
#include <stdio.h>

__device__ double caculateValueOfWeight(double parameter, int sign, double alpha)
{
	return (parameter*sign*alpha);
}

__global__ void updateWeights(double* weights, double* parameters,double* otherp, int sign, double alpha)
{
	int index = threadIdx.x;
	double value = weights[index];
	weights[index] = value + caculateValueOfWeight( parameters[index], sign, alpha);
	
}

cudaError_t updateWeightsWithCuda(double * weights, double * parameters, double * alpha, int * sign, int dimensionSize)
{
	cudaError_t cudaStatus;

	double* weightsOnGPU;
	double* parametersOnGPU;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return cudaErrorUnknown;
	}

	cudaStatus = cudaMalloc((void**)&weightsOnGPU, (dimensionSize + 1) * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc of weights failed");
		return cudaErrorUnknown;
	}
	cudaStatus = cudaMalloc((void**)&parametersOnGPU, (dimensionSize + 1) * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc of parameters failed");
		return cudaErrorUnknown;
	}

	cudaStatus = cudaMemcpy(weightsOnGPU, weights, (dimensionSize + 1) * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CudaMemCpy  of weights failed");
		return cudaErrorUnknown;
	}

	cudaStatus = cudaMemcpy(parametersOnGPU, parameters, (dimensionSize + 1) * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CudaMemCpy  of weights failed");
		return cudaErrorUnknown;
	}

	updateWeights << <1, dimensionSize + 1 >> > (weightsOnGPU, parametersOnGPU,parameters, *sign, *alpha);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Update  of weights failed");
		return cudaErrorUnknown;
	}

	cudaStatus = cudaMemcpy(weights, weightsOnGPU, (dimensionSize + 1)*sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CudaMemCpy  of weights back failed");
		return cudaErrorUnknown;
	}

	
	return cudaStatus;
}

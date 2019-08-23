
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <mpi.h>
#include <math.h>
#include "MyMacro.h"
#include "MyProto.h"
#pragma warning(disable: 4996)

void readDataFromFile(double **pointParameters,int** labels, int *numOfPoints, int *dimensionSize, int *LIMIT, double *QC, double *alphaZero, double *alphaMax)
{
	FILE * fp;
	fp = fopen(PATH, "r");
	if (fp == NULL)
	{
		fprintf(stderr,"Could not open data file");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	fscanf(fp, "%d %d %lf %lf %d %lf\n", numOfPoints, dimensionSize, alphaZero, alphaMax, LIMIT, QC);
	*pointParameters = (double*)malloc((*dimensionSize) * sizeof(double) * (*numOfPoints));
	*labels = (int*)malloc((*numOfPoints) * sizeof(int));
	for (int i = 0; i < *numOfPoints; i++)
	{
		for (int j = 0; j < *dimensionSize; j++)
		{
			fscanf(fp,"%lf ", &((*pointParameters)[i*(*dimensionSize) + j]));
		}
		fscanf(fp, "%d\n", &((*labels)[i]));
	}
	fclose(fp);
}

void sendDataToProcess(int myrank, double **pointParameters, int **labels, int *numOfPoints, int *dimensionSize, int *LIMIT, double *QC, double *alphaZero, double *alphaMax)
{
	MPI_Bcast(numOfPoints, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(dimensionSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	if (myrank != 0)
	{
		*pointParameters = (double*)malloc((*dimensionSize) * (*numOfPoints) * sizeof(double));
		*labels = (int*)malloc((*numOfPoints) * sizeof(int));
	}
	
	MPI_Bcast(*pointParameters, (*dimensionSize) *(*numOfPoints), MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(*labels, *numOfPoints, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(LIMIT, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(QC, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(alphaZero, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(alphaMax, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void resetWeights(double *weights, int dimensionSize)
{
	#pragma omp parallel for
	for (int i = 0; i < dimensionSize + 1; i++)
		weights[i] = 0;
	
}

void allocateAlphasToArray(double* alphasArray, double alphaZero,double  alphaMax, int numOfAlphasForProcess, int myrank, int size)
{
	int i;
	double tmpAlpha=alphaZero*(myrank+1);
	for (i = 0; i < numOfAlphasForProcess; i++)
	{
		
		if (tmpAlpha > alphaZero + alphaMax)
			alphasArray[i] = 0;
		else
			alphasArray[i] = tmpAlpha;
		
	
		tmpAlpha += (alphaZero * size);

		
	}

}

double caculateFunctionForNmis(double* weights, double* Points, int dimensionSize,int pointNumber)
{
	double result = 0;
	int i;
	for (i = 0; i < dimensionSize; i++)
	{
		result += weights[i] * Points[pointNumber*dimensionSize + i];
	}
	result += weights[dimensionSize];
	return result;
}

double caculateFunction(double* weights, double* parameters, int dimensionSize)
{
	int i;
	double function=0;
	for (i = 0; i < dimensionSize; i++)
	{
		function += weights[i] * parameters[i];
	}
	function += weights[dimensionSize];
	return function;
}

int caculateSign(double function)
{
	if (function >= 0)
		return 1;
	return -1;
}
void perceptronAlgorithm(double* alpha, double* alphasArray, int numOfAlphasForProcess, double* pointParameters, int* labels, double* weights, int numOfPoints, int dimensionSize, int LIMIT, double QC)
{
	int i, j, k;
	int AllPointsAreGood = 0, numOfIterations = 0, isBad = 0, Nmis = 0, sign = 0;
	cudaError_t cudaStatus;
	double q = 0;
	double* parameters = (double*)calloc(dimensionSize + 1, sizeof(double));
	parameters[dimensionSize] = 1;


	for (i = 0; i < numOfAlphasForProcess; i++)
	{

		//reset weights for next alpha
		resetWeights(weights, dimensionSize);
		Nmis = 0;
		if (alphasArray[i] == 0)
			break;
		numOfIterations = 0;
		AllPointsAreGood = 0;
		while (numOfIterations < LIMIT && AllPointsAreGood == 0)
		{
			isBad = 0;
			//caculate function
			for (j = 0; j < numOfPoints; j++)
			{
				#pragma omp parallel for
				for (k = 0; k < dimensionSize; k++)
					parameters[k] = pointParameters[(j*dimensionSize) + k];
				sign = caculateSign(caculateFunction(weights,parameters,dimensionSize));

				if (sign != labels[j])
				{
					isBad = 1;
					//Update weights with Cuda
					cudaStatus = updateWeightsWithCuda(weights, parameters, &alphasArray[i], &sign, dimensionSize);
					if (cudaStatus != cudaSuccess)
					{
						fprintf(stderr, "Update of Weights Failed!");
						MPI_Abort(MPI_COMM_WORLD, 1);
					}
					numOfIterations++;
					break;
				}
			}
			if (isBad == 0)
				AllPointsAreGood = 1;
		}
		if (AllPointsAreGood == 1)
			Nmis = 0;
		else
		{
			//caculate Nmis
			double function = 0;
			#pragma omp parallel for  private(sign,function) reduction(+:Nmis)
			for (j = 0; j < numOfPoints; j++)
			{	
				function = caculateFunctionForNmis(weights, pointParameters, dimensionSize,j);
				sign = caculateSign(function);
				if (sign != labels[j])
					Nmis++;
			}
		}
		q = (double)Nmis / (double)numOfPoints;
		if (q < QC)
		{
			
			*alpha = alphasArray[i];
			return;
		}


	}
}

void collectAlphasAndWeights(double* AllAlphas, double** AllWeights, double alpha, double* weights, int dimensionSize,int myrank,int size)
{
	int i,j;
	MPI_Status status;
	if (myrank == 0)
	{
		AllAlphas[myrank] = alpha;
		for (i = 0; i < dimensionSize + 1; i++)
		{
			AllWeights[myrank][i] = weights[i];
		}
		for (i = 1; i < size; i++)
		{
			MPI_Recv(&AllAlphas[i], 1, MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			for (j = 0; j < dimensionSize+1; j++)
			{
				MPI_Recv(&AllWeights[i][j], 1, MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			}

		}
	}
	else
	{
		MPI_Send(&alpha, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		for (i = 0; i < dimensionSize+1; i++)
		{
			MPI_Send(&weights[i], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		}
	}
	
}

void findMinAlpha(double* AllAlphas, double** AllWeights, int dimensionSize,int size)
{
	int i,minIndex;
	double minAlpha=0;
	FILE * fp;
	fp = fopen(OUTPUT, "w");
	if (fp == NULL)
	{
			fprintf(stderr, "Could not open output file");
			MPI_Abort(MPI_COMM_WORLD, 1);
	}
	for (i = 0; i < size; i++)
	{
		if (AllAlphas[i] != 0 && minAlpha == 0)
		{
			minAlpha = AllAlphas[i];
			minIndex = i;
		}
		if (AllAlphas[i] < minAlpha)
		{
			minAlpha = AllAlphas[i];
			minIndex = i;
		}
	}
	if (minAlpha == 0)
	{
		fprintf(fp, "No alpha Found!");
	}
	else
	{
		fprintf(fp,"minimum alpha is---->%lf\n", AllAlphas[minIndex]);
		for (i = 0; i < dimensionSize + 1; i++)
			fprintf(fp,"w%d is--->%lf\n",i,AllWeights[minIndex][i]);
	}
	printf("result in output File\n");
}
int main(int argc, char *argv[])
{
	int myrank, size;
	int i;
	int numOfPoints, dimensionSize, LIMIT, *labels;
	double alphaZero, alphaMax, QC;
	double * weights;
	double *pointParameters;
	int numOfAlphasForProcess = 0;
	double alpha = 0;
	double* alphasArray;
	double* functionArray;
	double* AllAlphas;
	double** AllWeights;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	double t1 = MPI_Wtime();
	if (myrank == 0)
		readDataFromFile(&pointParameters, &labels, &numOfPoints, &dimensionSize, &LIMIT, &QC, &alphaZero, &alphaMax);
	sendDataToProcess(myrank, &pointParameters, &labels, &numOfPoints, &dimensionSize, &LIMIT, &QC, &alphaZero, &alphaMax);
	numOfAlphasForProcess = int(ceil(alphaMax / alphaZero/size));
	//printf("%d\n", numOfAlphasForProcess);
	alphasArray = (double*)calloc(numOfAlphasForProcess, sizeof(double));
	allocateAlphasToArray(alphasArray, alphaZero, alphaMax, numOfAlphasForProcess, myrank, size);
	if (myrank == 0)
	{
		AllAlphas = (double*)calloc(size, sizeof(double));;
		AllWeights = (double**)malloc(size * sizeof(double*));;
		for (i = 0; i < dimensionSize + 1; i++)
		{
			AllWeights[i] = (double*)calloc(dimensionSize + 1, sizeof(double));
		}
	}
	weights = (double*)calloc(dimensionSize + 1, sizeof(double));
	perceptronAlgorithm(&alpha, alphasArray ,numOfAlphasForProcess,  pointParameters, labels,weights, numOfPoints, dimensionSize, LIMIT, QC);
	collectAlphasAndWeights(AllAlphas, AllWeights, alpha, weights, dimensionSize,myrank,size);
	
	if (myrank == 0)
	{
		findMinAlpha(AllAlphas, AllWeights, dimensionSize,size);
		double t2 = MPI_Wtime();
		printf("time was %e\n", (t2 - t1));
	}
	
	MPI_Finalize();

	return 0;
}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
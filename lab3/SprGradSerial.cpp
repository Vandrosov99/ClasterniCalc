#include <stdio.h>
#include <stdlib.h>
#include <conio.h>
#include <time.h>
#include <math.h>

int* pSerialPivotPos;  // Number of pivot rows selected at the iterations
int* pSerialPivotIter; // Iterations, at which the rows were pivots

// Function for formatted vector output
void PrintVector(double* pVector, int Size) {
	int i;
	for (i = 0; i < Size; i++)
		printf("%7.4f ", pVector[i]);
}
// Function for simple initialization of the matrix and the vector elements
void DummyDataInitialization(double* pMatrix, double* pVector, int Size) {
	int i, j;
	for (i = 0; i < Size; i++) {
		pVector[i] = rand() / double(1000);
		for (j = 0; j < Size; j++) {
			//if (j <= i)
			pMatrix[i*Size + j] = 1;
			//else
				//pMatrix[i*Size + j] = 0;
		}
	}
}
// Function for formatted matrix output
void PrintMatrix(double* pMatrix, int RowCount, int ColCount) {
	int i, j; // Loop variables
	for (i = 0; i < RowCount; i++) {
		for (j = 0; j < ColCount; j++)
			printf("%7.4f ", pMatrix[i*RowCount + j]);
		printf("\n");
	}
}
// Function for random initialization of the matrix and the vector elements
void RandomDataInitialization(double* pMatrix, double* pVector, int Size) {
	int i, j,k;  // Loop variables
	srand(unsigned(clock()));

	/*pMatrix[0] = 3;
	pMatrix[1] = -1;
	pMatrix[2] = -1;
	pMatrix[3] = 3;
	pVector[0] = 3;
	pVector[1] = 7;*/
	double* B = new double[Size*Size];
	for (i = 0; i < Size; i++) {
		pVector[i] = rand() / double(1000);
		for (j = 0; j < Size; j++) {
			//if (j <= i)
			B[i*Size + j] = rand() / double(1000);
			pMatrix[i*Size + j] = 0;
			//else
				//pMatrix[i*Size + j] = 0;
		}
	}

	int i1, j1, k1;
		//PrintMatrix(B, Size, Size);
		for (i1 = 0; i1 < Size; i1++) {
			for (j1 = 0; j1 < Size; j1++)
				for (k1 = 0; k1 < Size; k1++)
					pMatrix[i1*Size + j1] += B[k1*Size + i1] * B[k1*Size + j1];
		}
		//PrintMatrix(pMatrix, Size, Size);
		//PrintVector(pVector, Size);
		delete[] B;
}

// Function for memory allocation and data initialization
void ProcessInitialization(double* &pMatrix, double* &pVector,
	double* &pResult, int &Size, double* &pGradient, double* &pD, double &pS, double* &pXk, 
	double* &pXkminus1, double* &pGradientKminus1, double* &pDkminus1) {
	// Setting the size of the matrix and the vector
	/*do {
		printf("\nEnter size of the matrix and the vector: ");
		scanf_s("%d", &Size);
		printf("\nChosen size = %d \n", Size);

		if (Size <= 0)
			printf("\nSize of objects must be greater than 0!\n");
	} while (Size <= 0);*/

	// Memory allocation 
	pMatrix = new double[Size*Size];
	pVector = new double[Size];
	pResult = new double[Size];
	pGradient = new double[Size];
	pD = new double[Size];
	pXk = new double[Size];
	pXkminus1 = new double[Size];
	pGradientKminus1 = new double[Size];
	pDkminus1 = new double[Size];
	
	// Initialization of the matrix and the vector elements
	//DummyDataInitialization(pMatrix, pVector, Size);
	RandomDataInitialization(pMatrix, pVector, Size);

	int i;
	for (i = 0; i < Size; i++) {
		pXkminus1[i] = 0;
		pDkminus1[i] = 0;
		pGradientKminus1[i] = 0 - pVector[i];
	}
}




double ScalarDobutok(double* a, double* b, int Size) {
	int i;

	double res = 0;
	for (i = 0; i < Size; i++) {
		res += a[i] * b[i];
	}
	return res;
}

//Grad - Step 1
void SerialGradientCalculation(double* pMatrix, double* pVector, double* pX, double* pGradient, int Size, 
	double* pXkminus1, double* pGradientKminus1) {
	int i, j;  // Loop variables
	for (i = 0; i < Size; i++) {
		pGradient[i] = 0;
		for (j = 0; j < Size; j++)
			pGradient[i] += pMatrix[i*Size + j] * pXkminus1[j];//j
		pGradient[i] -= pVector[i];
	}
	//printf_s("g is: ");
	//PrintVector(pGradient, Size);
}
//D - Step 2
void SerialDCalculation(double* pGradient, int Size, double* pGradientKminus1,
	double* pDkminus1, double* pD) {
	int i;
	double scalarGk;
	double scalarGkminus1;

	scalarGk = ScalarDobutok(pGradient, pGradient, Size);
	scalarGkminus1 = ScalarDobutok(pGradientKminus1, pGradientKminus1, Size);
	
	for (i = 0; i < Size; i++) {
		pD[i] = 0 - pGradient[i] + (scalarGk /
			scalarGkminus1) * pDkminus1[i];
		pDkminus1[i] = pD[i];
	}
	//printf_s("d is: ");
	//PrintVector(pD, Size);
}
//S - Step 3
double SerialSCalculation(double* pMatrix,
	int Size, double* pGradient, double* pD) {
	double scalarD_G;
	scalarD_G = ScalarDobutok(pD, pGradient, Size);
	int i,j;
	double* temp = new double[Size];//d^k ^T * A
	for (i = 0; i < Size; i++) {
		temp[i] = 0;
		for (j = 0; j < Size; j++) {
			temp[i] += pGradient[j] * pMatrix[i*Size + j];
		}
	}
	double scalarTemp_D;
	scalarTemp_D = ScalarDobutok(temp, pD, Size);
	double pS;
	pS = scalarD_G / scalarTemp_D;
	//printf("%7.4f ", pS);
	//printf_s("\n");
	return pS;
}

//step 4
void SerialXCalculation(double* pD, double pS, double* pXkminus1, int Size, double* pResult) {
	int i;
	for (i = 0; i < Size; i++) {
		pResult[i] = pXkminus1[i] + pS * pD[i];
		pXkminus1[i] = pResult[i];
	}
	//printf_s("x is: ");
	//(pResult, Size);
}
// Function for the execution of Gauss algorithm
void SerialResultCalculation(double* pMatrix, double* pVector,
	double* pResult, int Size, double* pGradient, double* pD, double pS, double* pXk,
	double* pXkminus1, double* pGradientKminus1, double* pDkminus1) {

	for (int iter = 0; iter < Size; iter++) {
		//printf_s("iter " + iter);
		SerialGradientCalculation(pMatrix, pVector, pXk, pGradient, Size, pXkminus1, pGradientKminus1);
		SerialDCalculation(pGradient, Size, pGradientKminus1, pDkminus1, pD);
		double s = SerialSCalculation(pMatrix,
			Size,pGradient, pD);
		SerialXCalculation(pD, s, pXkminus1, Size, pResult);
		for (int i = 0; i < Size; i++) {
			pGradientKminus1[i] = pGradient[i];

		}
	}
}

// Function for computational process termination
void ProcessTermination(double* pMatrix, double* pVector, double* pResult,
	double* pGradient, double* pD, double* pXk,
	double* pXkminus1, double* pGradientKminus1, double* pDkminus1) {
	delete[] pMatrix;
	delete[] pVector;
	delete[] pResult;
	delete[] pGradient;
	delete[] pD;
	delete[] pXk;
	delete[] pXkminus1;
	delete[] pGradientKminus1;
	delete[] pDkminus1;
}


void main() {
	double* pMatrix;  // Matrix of the linear system
	double* pVector;  // Right parts of the linear system
	double* pResult;  // Result vector
	int Size;         // Size of the matrix and the vector

	double* pGradient; //
	double* pD;// вектор напрямку
	double pS; // величина зміщення
	double* pXk; ///чергове наближення

	double* pXkminus1;
	double* pGradientKminus1;
	double* pDkminus1;

	time_t start, finish;
	double duration;
	int N[] = { 10,100, 500, 1000, 1500, 2000,2500,3000 };
	for (int i = 0; i < 8; i++) {
		Size = N[i];
		printf("Serial Gradient algorithm for solving linear systems\n");
		// Memory allocation and definition of objects' elements
		ProcessInitialization(pMatrix, pVector, pResult, Size, pGradient, pD, pS, pXk,
			pXkminus1, pGradientKminus1, pDkminus1);

		// The matrix and the vector output
		//printf("Initial Matrix \n");
		//PrintMatrix(pMatrix, Size, Size);
		//printf("Initial Vector \n");
		//PrintVector(pVector, Size);

		// Execution of Gauss algorithm
		start = clock();
		SerialResultCalculation(pMatrix, pVector, pResult, Size, pGradient, pD, pS, pXk,
			pXkminus1, pGradientKminus1, pDkminus1);
		finish = clock();
		duration = (finish - start) / double(CLOCKS_PER_SEC);

		// Printing the result vector
		//printf("\n Result Vector: \n");
		//PrintVector(pResult, Size);

		// Printing the execution time of Gauss method
		printf("\n Time of execution: %f\n", duration);

		// Computational process termination
		ProcessTermination(pMatrix, pVector, pResult, pGradient, pD, pXk,
			pXkminus1, pGradientKminus1, pDkminus1);
	}
	getchar();
}

#include <stdio.h>
#include <stdlib.h>
#include <conio.h>
#include <time.h>
#include <mpi.h>

int ProcNum = 0;      // Number of available processes 
int ProcRank = 0;     // Rank of current process
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
	int i, j, k;  // Loop variables
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
	double* &pXkminus1, double* &pGradientKminus1, double* &pDkminus1,
	double* &pProcRows, double* &pProcResult, double* &pPartialGradient, int &RowNum) {
	// Setting the size of the matrix and the vector
	/*do {
		printf("\nEnter size of the matrix and the vector: ");
		scanf_s("%d", &Size);
		printf("\nChosen size = %d \n", Size);

		if (Size <= 0)
			printf("\nSize of objects must be greater than 0!\n");
	} while (Size <= 0);*/
	int RestRows; // Number of rows, that haven’t been distributed yet
	int i;             // Loop variable
	setvbuf(stdout, 0, _IONBF, 0);
	MPI_Bcast(&Size, 1, MPI_INT, 0, MPI_COMM_WORLD);

	// Determine the number of matrix rows stored on each process
	RestRows = Size;
	for (i = 0; i < ProcRank; i++)
		RestRows = RestRows - RestRows / (ProcNum - i);
	RowNum = RestRows / (ProcNum - ProcRank);
	// Memory allocation 
	
	pVector = new double[Size];
	pResult = new double[Size];
	pGradient = new double[Size];
	pD = new double[Size];
	pXk = new double[Size];
	pXkminus1 = new double[Size];
	pGradientKminus1 = new double[Size];
	pDkminus1 = new double[Size];
	pProcRows = new double[RowNum*Size];
	pProcResult = new double[RowNum];
	pPartialGradient = new double[Size];

	if (ProcRank == 0) {
		pMatrix = new double[Size*Size];
		RandomDataInitialization(pMatrix, pVector, Size);

	}
	// Initialization of the matrix and the vector elements
	//DummyDataInitialization(pMatrix, pVector, Size);

	int k;
	for (k = 0; k < Size; k++) {
		pXkminus1[k] = 0;
		pDkminus1[k] = 0;
		pGradientKminus1[k] = 0 - pVector[k];
	}
	MPI_Bcast(pXkminus1, Size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

}
// Function for distribution of the initial objects between the processes
void DataDistribution(double* pMatrix, double* pProcRows, double* pVector,
	int Size, int RowNum) {
	int *pSendNum; // The number of elements sent to the process
	int *pSendInd; // The index of the first data element sent to the process
	int RestRows = Size; // Number of rows, that haven’t been distributed yet

	MPI_Bcast(pVector, Size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// Alloc memory for temporary objects
	pSendInd = new int[ProcNum];
	pSendNum = new int[ProcNum];

	// Define the disposition of the matrix rows for current process
	RowNum = (Size / ProcNum);
	pSendNum[0] = RowNum * Size;
	pSendInd[0] = 0;
	for (int i = 1; i < ProcNum; i++) {
		RestRows -= RowNum;
		RowNum = RestRows / (ProcNum - i);
		pSendNum[i] = RowNum * Size;
		pSendInd[i] = pSendInd[i - 1] + pSendNum[i - 1];
	}

	// Scatter the rows
	MPI_Scatterv(pMatrix, pSendNum, pSendInd, MPI_DOUBLE, pProcRows,
		pSendNum[ProcRank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// Free the memory
	if(pSendNum!=NULL)
	delete[] pSendNum;
	delete[] pSendInd;
}

void VectorDistribution(double* pMatrix, double* pProcRows, double* pVector,
	int Size, int RowNum) {
	int *pSendNum; // The number of elements sent to the process
	int *pSendInd; // The index of the first data element sent to the process
	int RestRows = Size; // Number of rows, that haven’t been distributed yet

	MPI_Bcast(pVector, Size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// Alloc memory for temporary objects
	pSendInd = new int[ProcNum];
	pSendNum = new int[ProcNum];

	// Define the disposition of the matrix rows for current process
	RowNum = (Size / ProcNum);
	pSendNum[0] = RowNum * Size;
	pSendInd[0] = 0;
	for (int i = 1; i < ProcNum; i++) {
		RestRows -= RowNum;
		RowNum = RestRows / (ProcNum - i);
		pSendNum[i] = RowNum * Size;
		pSendInd[i] = pSendInd[i - 1] + pSendNum[i - 1];
	}

	// Scatter the rows
	MPI_Scatterv(pMatrix, pSendNum, pSendInd, MPI_DOUBLE, pProcRows,
		pSendNum[ProcRank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// Free the memory
	delete[] pSendNum;
	delete[] pSendInd;
}
void TestDistribution(double* pMatrix, double* pVector, double* pProcRows,
	int Size, int RowNum) {
	if (ProcRank == 0) {
		printf("Initial Matrix: \n");
		PrintMatrix(pMatrix, Size, Size);
		printf("Initial Vector: \n");
		PrintVector(pVector, Size);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	for (int i = 0; i < ProcNum; i++) {
		if (ProcRank == i) {
			printf("\nProcRank = %d \n", ProcRank);
			printf(" Matrix Stripe:\n");
			PrintMatrix(pProcRows, RowNum, Size);
			printf(" Vector: \n");
			PrintVector(pVector, Size);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
}
// Result vector replication
void ResultReplication(double* pProcResult, double* pPartialGradient, int Size,
	int RowNum) {
	int i;             // Loop variable
	int *pReceiveNum;  // Number of elements, that current process sends
	int *pReceiveInd;  /* Index of the first element from current process
						  in result vector */
	int RestRows = Size; // Number of rows, that haven’t been distributed yet

	//Alloc memory for temporary objects
	pReceiveNum = new int[ProcNum];
	pReceiveInd = new int[ProcNum];

	//Define the disposition of the result vector block of current processor
	pReceiveInd[0] = 0;
	pReceiveNum[0] = Size / ProcNum;
	for (i = 1; i < ProcNum; i++) {
		RestRows -= pReceiveNum[i - 1];
		pReceiveNum[i] = RestRows / (ProcNum - i);
		pReceiveInd[i] = pReceiveInd[i - 1] + pReceiveNum[i - 1];
	}

	//Gather the whole result vector on every processor
	MPI_Allgatherv(pProcResult, pReceiveNum[ProcRank], MPI_DOUBLE, pPartialGradient,
		pReceiveNum, pReceiveInd, MPI_DOUBLE, MPI_COMM_WORLD);

	//Free the memory
	delete[] pReceiveNum;
	delete[] pReceiveInd;
}


double ScalarDobutok(double* a, double* b, int Size) {
	int i;

	double res = 0;
	for (i = 0; i < Size; i++) {
		res += a[i] * b[i];
	}
	return res;
}
// Process rows and vector multiplication
void ParallelMatrixVectorMult(double* pProcRows, double* pVector, double* pProcResult, int Size, int RowNum) {
	int i, j;  // Loop variables
	for (i = 0; i < RowNum; i++) {
		pProcResult[i] = 0;
		for (j = 0; j < Size; j++)
			pProcResult[i] += pProcRows[i*Size + j] * pVector[j];
	}
}
//Grad - Step 1
void GradientCalculation(double* pMatrix, double* pVector, double* pX, double* pGradient, int Size,
	double* pXkminus1, double* pPartialGradient) {
	int i, j;  // Loop variables
	for (i = 0; i < Size; i++) {		
		pGradient[i] = pPartialGradient[i] - pVector[i];
	}
	//printf_s("g is: ");
	//PrintVector(pGradient, Size);
}
//D - Step 2
void DCalculation(double* pGradient, int Size, double* pGradientKminus1,
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
double SCalculation(double* pMatrix,
	int Size, double* pGradient, double* pD) {
	double scalarD_G;
	scalarD_G = ScalarDobutok(pD, pGradient, Size);
	int i, j;
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
void XCalculation(double* pD, double pS, double* pXkminus1, int Size, double* pResult) {
	int i;
	for (i = 0; i < Size; i++) {
		pResult[i] = pXkminus1[i] + pS * pD[i];
		pXkminus1[i] = pResult[i];
	}
	MPI_Bcast(pXkminus1, Size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	//printf_s("x is: ");
	//(pResult, Size);
}
// Function for the execution of Gauss algorithm
void ParallelResultCalculation(double* pMatrix, double* pVector,
	double* pResult, int Size, double* pGradient, double* pD, double pS, double* pXk,
	double* pXkminus1, double* pGradientKminus1, double* pDkminus1, double* pPartialGradient,
	double* pProcRows, double* pProcResult, int RowNum) {

	for (int iter = 0; iter < Size; iter++) {
		//printf_s("iter " + iter);

		ParallelMatrixVectorMult(pProcRows, pXkminus1, pProcResult,Size, RowNum);

		// Result replication
		ResultReplication(pProcResult, pPartialGradient, Size, RowNum);
		if (ProcRank == 0) {
			GradientCalculation(pMatrix, pVector, pXk, pGradient, Size, pXkminus1, pPartialGradient);
			DCalculation(pGradient, Size, pGradientKminus1, pDkminus1, pD);
			double s = SCalculation(pMatrix,
				Size, pGradient, pD);
			XCalculation(pD, s, pXkminus1, Size, pResult);
			for (int i = 0; i < Size; i++) {
				pGradientKminus1[i] = pGradient[i];

			}
		}
		MPI_Barrier(MPI_COMM_WORLD);

	}
}

// Function for computational process termination
void ProcessTermination(double* pMatrix, double* pVector, double* pResult,
	double* pGradient, double* pD, double* pXk,
	double* pXkminus1, double* pGradientKminus1, double* pDkminus1, double* pPartialGradient,
	double* pProcRows, double* pProcResult) {
	delete[] pMatrix;
	delete[] pVector;
	delete[] pResult;
	delete[] pGradient;
	delete[] pD;
	delete[] pXk;
	delete[] pXkminus1;
	delete[] pGradientKminus1;
	delete[] pDkminus1;
	delete[] pPartialGradient;
	delete[] pProcRows;
	delete[] pProcResult;

}


void main(int argc, char* argv[]) {
	double* pMatrix;  // Matrix of the linear system
	double* pVector;  // Right parts of the linear system
	double* pResult;  // Result vector
	int Size;         // Size of the matrix and the vector

	double* pGradient; //
	double* pD;// âåêòîð íàïðÿìêó
	double pS; // âåëè÷èíà çì³ùåííÿ
	double* pXk; ///÷åðãîâå íàáëèæåííÿ

	double* pXkminus1;
	double* pGradientKminus1;
	double* pDkminus1;

	double* pPartialGradient;
	double* pProcRows;   // Stripe of the matrix on the current process
	double* pProcResult; // Block of the result vector on the current process
	int RowNum;          // Number of rows in the matrix stripe
	time_t start, finish;
	double duration;


	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
	MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
	if(ProcRank==0)
		getchar();
	int N[] = { 10,100, 500, 1000, 1500, 2000,2500,3000 };
	for (int i = 0; i < 8; i++) {
		Size = N[i];
		if (ProcRank == 0) {
			printf("Parallel Gradient algorithm for solving linear systems\n");
		}
		// Memory allocation and definition of objects' elements
		ProcessInitialization(pMatrix, pVector, pResult, Size, pGradient, pD, pS, pXk,
			pXkminus1, pGradientKminus1, pDkminus1, pProcRows, pProcResult,
			pPartialGradient, RowNum);

		// The matrix and the vector output
		//printf("Initial Matrix \n");
		//PrintMatrix(pMatrix, Size, Size);
		//printf("Initial Vector \n");
		//PrintVector(pVector, Size);

		// Distributing the initial objects between the processes
		DataDistribution(pMatrix, pProcRows, pVector, Size, RowNum);

		// Distribution test
		//TestDistribution(pMatrix, pVector, pProcRows, Size, RowNum);

		double Start = MPI_Wtime();


		ParallelResultCalculation(pMatrix, pVector, pResult, Size, pGradient, pD, pS, pXk,
			pXkminus1, pGradientKminus1, pDkminus1,pPartialGradient, pProcResult,
			pPartialGradient, RowNum);

		

		double Finish = MPI_Wtime();
		double Duration = Finish - Start;

		// Printing the result vector
		//printf("\n Result Vector: \n");
		//PrintVector(pResult, Size);

		if (ProcRank == 0) {
			printf("Time of execution = %f\n", Duration);
			//printf("N = %d ; Time of execution = %f\n",N[i], Duration);
		}

		// Computational process termination
		ProcessTermination(pMatrix, pVector, pResult, pGradient, pD, pXk,
			pXkminus1, pGradientKminus1, pDkminus1, pPartialGradient, pProcRows, pProcResult);
	}
	MPI_Finalize();

	//getchar();
}


#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#define Nx 16384
#define Ny 16384


double cpuTime(){
	struct timeval tp;
	gettimeofday(&tp,NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-06);
}


void initialMatrixZ(float *h_Mat, int size){
	float *matrix = h_Mat;
	for (int i = 0; i < size; ++i){
		matrix[i] = 0.0;
	}
}

void initialMatrix0(float *h_Mat, int size){
	float *matrix = h_Mat;
	for (int i = 0; i < size; ++i){
		matrix[i] = 1.0;
	}
}

void printMatrix(float *h_Mat, int rows, int cols){
	float *h_Matrix = h_Mat;
	for(int i = 0; i < rows; ++i){
		for(int j = 0; j < cols; ++j){
			printf("%f ", h_Matrix[j]);
		}
		h_Matrix += cols;
		printf("\n");
	}
}

__global__ void sumaOnGPU(float *d_A, float *d_B, float *d_C, int width, int height){
	int col = (blockIdx.x * blockDim.x) + threadIdx.x; 
	int row = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	if (col >= width || row >= height) return; // acotamos el grid a el tamaño de la matriz de datos.

	int idx = col + (row * width);
	d_C[idx] = d_A[idx] + d_B[idx];
}

void sumaOnCPU(float *h_A, float *h_B, float *h_C, int rows, int cols){
	size_t size = rows * cols;
	for(int j = 0; j < size; j++){
		h_C[j] = h_A[j] + h_B[j]; 
	} 		
}


void compareResult(float *MA, float *MB, int size){
	double epsilon = 1e-8;
	bool flag = 1;
	for(int i=0; i<size; ++i){
		if (abs(MA[i] - MB[i]) > epsilon){
			printf("Suma distinta\n");
			printf("[%d]: MA = %f MB = %f\n",i,MA[i],MB[i]);
			flag = 0;
			break;
		}
	}
	if (flag){
		printf("Suma Correcta\n");
	}
}

int main(int argc, char const *argv[]){
	//Variables para medir el tiempo sobre el CPU
	double tic, toc, tictocC, tictocG = 0;
	
	int nElements = Nx*Ny;
	size_t nBytes = nElements * sizeof(float);
	float *h_A, *h_B, *h_C, *h_onGpu;

	h_A = (float*) malloc(nBytes);
	initialMatrix0(h_A, nElements);

	h_B = (float*) malloc(nBytes);
	initialMatrix0(h_B, nElements);

	h_C = (float*) malloc(nBytes);
	initialMatrixZ(h_C, nElements);

	h_onGpu = (float*) malloc(nBytes);
	initialMatrixZ(h_onGpu, nElements);


	// 1. Solicitamos espacio en la GPU.
	float *d_A, *d_B, *d_C;
	cudaMalloc((void**) &d_A, nBytes);
	cudaMalloc((void**) &d_B, nBytes);
	cudaMalloc((void**) &d_C, nBytes);

	// 2. Transferimos la informacion a la GPU
	cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_C, nBytes, cudaMemcpyHostToDevice);

	//Suma en CPU.
	tic = cpuTime();
	sumaOnCPU(h_A, h_B, h_C, Ny, Nx);
	toc = cpuTime();
	tictocC = toc - tic;

	// 3. Lanzamos el kernel
	int hilos_fila = 16;
	int hilos_columna = 16; 
	dim3 block(hilos_columna, hilos_fila);
	dim3 grid((Nx+ hilos_columna - 1)/hilos_columna , (Ny+ hilos_fila - 1)/hilos_fila);

	tic = cpuTime();
	sumaOnGPU<<<grid, block>>>(d_A, d_B, d_C, Nx, Ny);
	cudaDeviceSynchronize();
	toc = cpuTime();
	tictocG = toc - tic;
	
	// 4. Traer resultados de la GPU e imprimir informe del programa.

	cudaMemcpy(h_onGpu, d_C, nBytes, cudaMemcpyDeviceToHost);
	printf("----------------------------------\n");
	printf("Dimensiones de las matrices a procesar: Nx (columnas) = %d | Ny (filas) = %d\n", Nx, Ny);
	printf("La dimension del grid a usar es: grid(%u, %u)\n", grid.x, grid.y );
	printf("La dimension de los bloques es : block(%u, %u)\n", block.x, block.y); 
	printf("Tiempo de procesamiento de la CPU: %f\n", tictocC);
	printf("Tiempo de procesamiento de la GPU: %f\n", tictocG);
	printf("Speed-Up: %f\n",(tictocC/tictocG));
	compareResult(h_C, h_onGpu, nElements);
	printf("----------------------------------\n");

	// 5. Liberamos memoria
	free(h_A);
	free(h_B);
	free(h_C);
	free(h_onGpu);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

}

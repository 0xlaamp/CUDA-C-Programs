/* 
	Programa : Operacion Multiple Elemento a Elemento (FMA Vectorial) utilizando CUDA.

	Aplica una operacion combinada de multiplicacion y suma por cada elemento
	de manera independiente:

		D[i] = (A[i] * B[i]) + C[i]

	Cada elemento es independiente de los demas, por lo que cada hilo de la GPU
	se encarga de calcular un unico elemento del vector resultado.

	Elementos:

	A, B, C  ->  Vectores de entrada.
	D        ->  Vector resultado.

	Ejemplo:

	Entrada: A = [1, 2, 3], B = [4, 5, 6], C = [7, 8, 9]
	Salida:  D = [11, 18, 27]

	El programa incluye una version en CPU y una version para CUDA

*/

#include <stdio.h>
#include <sys/time.h>
#include <stdbool.h>
#include <math.h>
#include <stdlib.h>

/* funcion que calcula el tiempo que tarda una CPU */

double cpuTime(){
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-06);
}

/* funcion para inicializar un vector con valores aleatorios entre 0 y 1 */

void initData(float *vec, int size){
	for(int i = 0; i < size; i++){
		vec[i] = (float)rand() / (float)RAND_MAX;
	}
}

/* funcion para validar el resultado */

void checkResult(float *h_D, float *h_res_D, int size){
	bool match = true;
	float epsilon = 1.0E-5;
	for(int i = 0; i < size; i++){
		if(fabs(h_D[i] - h_res_D[i]) > epsilon){
			printf("Error: resultado distinto en la posicion %d (CPU: %f, GPU: %f)\n", i, h_D[i], h_res_D[i]);
			match = false;
			break;
		}
	}
	if(match){
		printf("Resultado correcto\n");
	}
}

/* funcion que muestra en pantalla los primeros elementos de un arreglo */

void mostrarVector(const char *nombre, float *vec, int n){
	printf("%s: [ ", nombre);
	for(int i = 0; i < n; i++){
		printf("%f ", vec[i]);
	}
	printf("... ]\n");
}

/* funcion FMA para CPU (secuencial) */

void fmaCPU(float *A, float *B, float *C, float *D, int size){
	for(int i = 0; i < size; i++){
		D[i] = (A[i] * B[i]) + C[i];
	}
}

/* funcion FMA para GPU */

__global__ void fmaGPU(float *A, float *B, float *C, float *D, int size){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx < size){
		D[idx] = (A[idx] * B[idx]) + C[idx];
	}
}


int main(void){

	//Paso 1: Reservar memoria en Host e inicializar los datos.
	double tic, toc, tictocS = 0, tictocP = 0;

	int size = 1 << 24;   // 16,777,216 elementos
	size_t bytes = size * sizeof(float);

	float *h_A = (float*) malloc(bytes);
	float *h_B = (float*) malloc(bytes);
	float *h_C = (float*) malloc(bytes);
	float *h_D = (float*) malloc(bytes);      // resultado CPU
	float *h_res = (float*) malloc(bytes);    // resultado GPU

	initData(h_A, size);
	initData(h_B, size);
	initData(h_C, size);

	//Paso 2: Reservar memoria en Device.
	float *d_A, *d_B, *d_C, *d_D;
	cudaMalloc((void**) &d_A, bytes);
	cudaMalloc((void**) &d_B, bytes);
	cudaMalloc((void**) &d_C, bytes);
	cudaMalloc((void**) &d_D, bytes);

	//Paso 3: Transferir los datos de Host to Device.
	cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_C, bytes, cudaMemcpyHostToDevice);

	//Paso 4.1: Lanzar la funcion serial de la CPU.
	tic = cpuTime();
	fmaCPU(h_A, h_B, h_C, h_D, size);
	toc = cpuTime();
	tictocS = toc - tic;

	//Paso 4.2: Lanzar la funcion paralela de la GPU.
	dim3 hilos(256);
	dim3 bloques((size + hilos.x - 1) / hilos.x);

	fmaGPU<<<bloques, hilos>>>(d_A, d_B, d_C, d_D, size); //calentamiento
	cudaDeviceSynchronize();

	tic = cpuTime();
	fmaGPU<<<bloques, hilos>>>(d_A, d_B, d_C, d_D, size);
	cudaDeviceSynchronize();
	toc = cpuTime();
	tictocP = toc - tic;

	//Paso 5: Transferir los resultados a la CPU.
	cudaMemcpy(h_res, d_D, bytes, cudaMemcpyDeviceToHost);

	// Mostrar resultados (solo los primeros elementos por el tamano del vector).
	mostrarVector("A      ", h_A, 5);
	mostrarVector("B      ", h_B, 5);
	mostrarVector("C      ", h_C, 5);
	mostrarVector("D (CPU)", h_D, 5);
	mostrarVector("D (GPU)", h_res, 5);
	printf("\n");

	// Verificamos resultados de la GPU y calculamos SpeedUP
	checkResult(h_D, h_res, size);
	printf("Tiempo de ejecucion en CPU: %f\n", tictocS);
	printf("Tiempo de ejecucion en GPU: %f\n", tictocP);
	printf("SpeedUp: %lf\n", tictocS/tictocP);

	//Paso 6: Liberar memoria de ambos lados.
	free(h_A);
	free(h_B);
	free(h_C);
	free(h_D);
	free(h_res);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaFree(d_D);

	return 0;
}

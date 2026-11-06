/* 
	Programa : Multiplicacion Escalar de una Matriz utilizando CUDA.

	Multiplica cada elemento de una matriz bidimensional por un factor escalar fijo
	utilizando coordenadas 2D (bloques e hilos en dos dimensiones).

		B[i][j] = alfa * A[i][j]

	La matriz se almacena linealmente en memoria (row-major), por lo que el
	elemento (fila, col) se encuentra en la posicion fila * columnas + col.

	Elementos:

	A     ->  Matriz de entrada.
	alfa  ->  Factor escalar.
	B     ->  Matriz resultado.

	Ejemplo:

	Entrada: A = [1 2]    alfa = 3
	             [3 4]
	Salida:  B = [3  6]
	             [9 12]

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

/* funcion para inicializar una matriz con valores aleatorios entre 0 y 1 */

void initData(float *mat, int filas, int columnas){
	for(int i = 0; i < filas * columnas; i++){
		mat[i] = (float)rand() / (float)RAND_MAX;
	}
}

/* funcion para validar el resultado */

void checkResult(float *h_B, float *h_res_B, int filas, int columnas){
	bool match = true;
	float epsilon = 1.0E-5;
	for(int i = 0; i < filas && match; i++){
		for(int j = 0; j < columnas; j++){
			int pos = i * columnas + j;
			if(fabs(h_B[pos] - h_res_B[pos]) > epsilon){
				printf("Error: resultado distinto en (%d, %d) (CPU: %f, GPU: %f)\n", i, j, h_B[pos], h_res_B[pos]);
				match = false;
				break;
			}
		}
	}
	if(match){
		printf("Resultado correcto\n");
	}
}

/* funcion que muestra en pantalla una esquina de la matriz (maximo n x n) */

void mostrarMatriz(const char *nombre, float *mat, int filas, int columnas, int n){
	printf("%s (esquina %dx%d):\n", nombre, n, n);
	for(int i = 0; i < n && i < filas; i++){
		printf("[ ");
		for(int j = 0; j < n && j < columnas; j++){
			printf("%8.4f ", mat[i * columnas + j]);
		}
		printf("... ]\n");
	}
	printf("...\n");
}

/* funcion multiplicacion escalar para CPU (secuencial) */

void escalarMatrizCPU(float *A, float *B, float alfa, int filas, int columnas){
	for(int i = 0; i < filas; i++){
		for(int j = 0; j < columnas; j++){
			B[i * columnas + j] = alfa * A[i * columnas + j];
		}
	}
}

/* funcion multiplicacion escalar para GPU (coordenadas 2D) */

__global__ void escalarMatrizGPU(float *A, float *B, float alfa, int filas, int columnas){
	int col  = blockDim.x * blockIdx.x + threadIdx.x;
	int fila = blockDim.y * blockIdx.y + threadIdx.y;

	if(fila < filas && col < columnas){
		int idx = fila * columnas + col;
		B[idx] = alfa * A[idx];
	}
}


int main(void){

	//Paso 1: Reservar memoria en Host e inicializar los datos.
	double tic, toc, tictocS = 0, tictocP = 0;

	int filas = 8192;
	int columnas = 8192;
	int size = filas * columnas;
	size_t bytes = size * sizeof(float);
	float alfa = 2.5f;

	float *h_A = (float*) malloc(bytes);
	float *h_B = (float*) malloc(bytes);      // resultado CPU
	float *h_res = (float*) malloc(bytes);    // resultado GPU

	initData(h_A, filas, columnas);

	//Paso 2: Reservar memoria en Device.
	float *d_A, *d_B;
	cudaMalloc((void**) &d_A, bytes);
	cudaMalloc((void**) &d_B, bytes);

	//Paso 3: Transferir los datos de Host to Device.
	cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);

	//Paso 4.1: Lanzar la funcion serial de la CPU.
	tic = cpuTime();
	escalarMatrizCPU(h_A, h_B, alfa, filas, columnas);
	toc = cpuTime();
	tictocS = toc - tic;

	//Paso 4.2: Lanzar la funcion paralela de la GPU.
	dim3 hilos(16, 16);
	dim3 bloques((columnas + hilos.x - 1) / hilos.x, (filas + hilos.y - 1) / hilos.y);

	escalarMatrizGPU<<<bloques, hilos>>>(d_A, d_B, alfa, filas, columnas); //calentamiento
	cudaDeviceSynchronize();

	tic = cpuTime();
	escalarMatrizGPU<<<bloques, hilos>>>(d_A, d_B, alfa, filas, columnas);
	cudaDeviceSynchronize();
	toc = cpuTime();
	tictocP = toc - tic;

	//Paso 5: Transferir los resultados a la CPU.
	cudaMemcpy(h_res, d_B, bytes, cudaMemcpyDeviceToHost);

	// Mostrar resultados (solo una esquina por el tamano de la matriz).
	printf("Factor escalar: %f\n\n", alfa);
	mostrarMatriz("Matriz original", h_A, filas, columnas, 4);
	mostrarMatriz("Resultado CPU", h_B, filas, columnas, 4);
	mostrarMatriz("Resultado GPU", h_res, filas, columnas, 4);
	printf("\n");

	// Verificamos resultados de la GPU y calculamos SpeedUP
	checkResult(h_B, h_res, filas, columnas);
	printf("Tiempo de ejecucion en CPU: %f\n", tictocS);
	printf("Tiempo de ejecucion en GPU: %f\n", tictocP);
	printf("SpeedUp: %lf\n", tictocS/tictocP);

	//Paso 6: Liberar memoria de ambos lados.
	free(h_A);
	free(h_B);
	free(h_res);
	cudaFree(d_A);
	cudaFree(d_B);

	return 0;
}

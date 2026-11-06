/* 
	Programa : Aplicacion de Funcion Seno/Coseno por Elemento utilizando CUDA.

	Calcula el seno y el coseno de cada elemento de un arreglo de angulos
	expresados en radianes, utilizando las funciones nativas de CUDA (sinf / cosf)
	en la GPU y sin() / cos() en la CPU.

	Cada angulo es independiente de los demas, por lo que cada hilo calcula
	el seno y el coseno de un unico elemento.

	Elementos:

	angulos  ->  Arreglo de entrada (radianes).
	seno     ->  Arreglo resultado con sin(angulos[i]).
	coseno   ->  Arreglo resultado con cos(angulos[i]).

	Ejemplo:

	Entrada: angulos = [0, 1.5708, 3.1416]
	Salida:  seno    = [0, 1, 0]
	         coseno  = [1, 0, -1]

	El programa incluye una version en CPU y una version para CUDA

*/

#include <stdio.h>
#include <sys/time.h>
#include <stdbool.h>
#include <math.h>
#include <stdlib.h>

#define PI 3.14159265358979f

/* funcion que calcula el tiempo que tarda una CPU */

double cpuTime(){
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-06);
}

/* funcion para inicializar los angulos con valores aleatorios entre 0 y 2*PI */

void initData(float *vec, int size){
	for(int i = 0; i < size; i++){
		vec[i] = 2.0f * PI * ((float)rand() / (float)RAND_MAX);
	}
}

/* funcion para validar el resultado (se usa una tolerancia por las diferencias
 * de precision entre sinf/cosf de la GPU y sin/cos de la CPU) */

void checkResult(float *h_ref, float *h_res, int size, const char *nombre){
	bool match = true;
	float epsilon = 1.0E-4;
	for(int i = 0; i < size; i++){
		if(fabs(h_ref[i] - h_res[i]) > epsilon){
			printf("Error en %s: distinto en la posicion %d (CPU: %f, GPU: %f)\n", nombre, i, h_ref[i], h_res[i]);
			match = false;
			break;
		}
	}
	if(match){
		printf("%s correcto\n", nombre);
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

/* funcion seno/coseno para CPU (secuencial) */

void senoCosenoCPU(float *angulos, float *seno, float *coseno, int size){
	for(int i = 0; i < size; i++){
		seno[i]   = sinf(angulos[i]);
		coseno[i] = cosf(angulos[i]);
	}
}

/* funcion seno/coseno para GPU */

__global__ void senoCosenoGPU(float *angulos, float *seno, float *coseno, int size){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx < size){
		seno[idx]   = sinf(angulos[idx]);
		coseno[idx] = cosf(angulos[idx]);
	}
}


int main(void){

	//Paso 1: Reservar memoria en Host e inicializar los datos.
	double tic, toc, tictocS = 0, tictocP = 0;

	int size = 1 << 24;   // 16,777,216 elementos
	size_t bytes = size * sizeof(float);

	float *h_ang = (float*) malloc(bytes);
	float *h_sen = (float*) malloc(bytes);       // seno CPU
	float *h_cos = (float*) malloc(bytes);       // coseno CPU
	float *h_res_sen = (float*) malloc(bytes);   // seno GPU
	float *h_res_cos = (float*) malloc(bytes);   // coseno GPU

	initData(h_ang, size);

	//Paso 2: Reservar memoria en Device.
	float *d_ang, *d_sen, *d_cos;
	cudaMalloc((void**) &d_ang, bytes);
	cudaMalloc((void**) &d_sen, bytes);
	cudaMalloc((void**) &d_cos, bytes);

	//Paso 3: Transferir los datos de Host to Device.
	cudaMemcpy(d_ang, h_ang, bytes, cudaMemcpyHostToDevice);

	//Paso 4.1: Lanzar la funcion serial de la CPU.
	tic = cpuTime();
	senoCosenoCPU(h_ang, h_sen, h_cos, size);
	toc = cpuTime();
	tictocS = toc - tic;

	//Paso 4.2: Lanzar la funcion paralela de la GPU.
	dim3 hilos(256);
	dim3 bloques((size + hilos.x - 1) / hilos.x);

	senoCosenoGPU<<<bloques, hilos>>>(d_ang, d_sen, d_cos, size); //calentamiento
	cudaDeviceSynchronize();

	tic = cpuTime();
	senoCosenoGPU<<<bloques, hilos>>>(d_ang, d_sen, d_cos, size);
	cudaDeviceSynchronize();
	toc = cpuTime();
	tictocP = toc - tic;

	//Paso 5: Transferir los resultados a la CPU.
	cudaMemcpy(h_res_sen, d_sen, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_res_cos, d_cos, bytes, cudaMemcpyDeviceToHost);

	// Mostrar resultados (solo los primeros elementos por el tamano del vector).
	mostrarVector("Angulos     ", h_ang, 5);
	mostrarVector("Seno (CPU)  ", h_sen, 5);
	mostrarVector("Seno (GPU)  ", h_res_sen, 5);
	mostrarVector("Coseno (CPU)", h_cos, 5);
	mostrarVector("Coseno (GPU)", h_res_cos, 5);
	printf("\n");

	// Verificamos resultados de la GPU y calculamos SpeedUP
	checkResult(h_sen, h_res_sen, size, "Seno");
	checkResult(h_cos, h_res_cos, size, "Coseno");
	printf("Tiempo de ejecucion en CPU: %f\n", tictocS);
	printf("Tiempo de ejecucion en GPU: %f\n", tictocP);
	printf("SpeedUp: %lf\n", tictocS/tictocP);

	//Paso 6: Liberar memoria de ambos lados.
	free(h_ang);
	free(h_sen);
	free(h_cos);
	free(h_res_sen);
	free(h_res_cos);
	cudaFree(d_ang);
	cudaFree(d_sen);
	cudaFree(d_cos);

	return 0;
}

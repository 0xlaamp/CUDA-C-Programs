/* 
	Programa : Aumento de Contraste Lineal de una imagen utilizando CUDA.

	Multiplica el valor de cada pixel de una imagen en escala de grises por un
	factor de contraste constante. Los valores se mantienen en el rango
	valido [0, 255] (si el resultado excede 255 se satura a 255).

		salida[i] = min(255, entrada[i] * factor)

	Una imagen en escala de grises es una matriz 2D donde cada pixel es un
	unsigned char (0 = negro, 255 = blanco). Se almacena linealmente en memoria
	(row-major) y se procesa con coordenadas 2D (un hilo por pixel).

	Elementos:

	imagen  ->  Imagen de entrada (pixeles de 0 a 255).
	factor  ->  Factor de contraste (> 1 aumenta, < 1 disminuye).
	salida  ->  Imagen resultado.

	Ejemplo:

	Entrada: pixeles = [10, 100, 200], factor = 1.5
	Salida:  pixeles = [15, 150, 255]    (200 * 1.5 = 300 -> se satura a 255)

	Nota: la imagen se genera de forma sintetica para no depender de librerias
	externas de lectura de imagenes.

	El programa incluye una version en CPU y una version para CUDA

*/

#include <stdio.h>
#include <sys/time.h>
#include <stdbool.h>
#include <stdlib.h>

/* funcion que calcula el tiempo que tarda una CPU */

double cpuTime(){
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-06);
}

/* funcion para inicializar una imagen con pixeles aleatorios entre 0 y 255 */

void initImagen(unsigned char *img, int alto, int ancho){
	for(int i = 0; i < alto * ancho; i++){
		img[i] = (unsigned char)(rand() % 256);
	}
}

/* funcion para validar el resultado */

void checkResult(unsigned char *h_ref, unsigned char *h_res, int alto, int ancho){
	bool match = true;
	for(int i = 0; i < alto && match; i++){
		for(int j = 0; j < ancho; j++){
			int pos = i * ancho + j;
			if(h_ref[pos] != h_res[pos]){
				printf("Error: pixel distinto en (%d, %d) (CPU: %d, GPU: %d)\n", i, j, h_ref[pos], h_res[pos]);
				match = false;
				break;
			}
		}
	}
	if(match){
		printf("Resultado correcto\n");
	}
}

/* funcion que muestra en pantalla una esquina de la imagen (maximo n x n) */

void mostrarImagen(const char *nombre, unsigned char *img, int alto, int ancho, int n){
	printf("%s (esquina %dx%d):\n", nombre, n, n);
	for(int i = 0; i < n && i < alto; i++){
		printf("[ ");
		for(int j = 0; j < n && j < ancho; j++){
			printf("%3d ", img[i * ancho + j]);
		}
		printf("... ]\n");
	}
	printf("...\n");
}

/* funcion aumento de contraste para CPU (secuencial) 
 *
 * Obs: el producto se calcula en float y se satura a 255 antes de regresar
 * 		a unsigned char, de lo contrario el valor daria la vuelta (overflow).
 */

void contrasteCPU(unsigned char *entrada, unsigned char *salida, float factor, int alto, int ancho){
	for(int i = 0; i < alto; i++){
		for(int j = 0; j < ancho; j++){
			int pos = i * ancho + j;
			float v = entrada[pos] * factor;
			if(v > 255.0f) v = 255.0f;
			if(v < 0.0f)   v = 0.0f;
			salida[pos] = (unsigned char)(v + 0.5f);   //redondeo al entero mas cercano
		}
	}
}

/* funcion aumento de contraste para GPU (coordenadas 2D, un hilo por pixel) */

__global__ void contrasteGPU(unsigned char *entrada, unsigned char *salida, float factor, int alto, int ancho){
	int col  = blockDim.x * blockIdx.x + threadIdx.x;
	int fila = blockDim.y * blockIdx.y + threadIdx.y;

	if(fila < alto && col < ancho){
		int pos = fila * ancho + col;
		float v = entrada[pos] * factor;
		if(v > 255.0f) v = 255.0f;
		if(v < 0.0f)   v = 0.0f;
		salida[pos] = (unsigned char)(v + 0.5f);
	}
}


int main(void){

	//Paso 1: Reservar memoria en Host e inicializar los datos.
	double tic, toc, tictocS = 0, tictocP = 0;

	int alto = 8192;
	int ancho = 8192;
	int size = alto * ancho;
	size_t bytes = size * sizeof(unsigned char);
	float factor = 1.5f;

	unsigned char *h_img = (unsigned char*) malloc(bytes);
	unsigned char *h_out = (unsigned char*) malloc(bytes);   // resultado CPU
	unsigned char *h_res = (unsigned char*) malloc(bytes);   // resultado GPU

	initImagen(h_img, alto, ancho);

	//Paso 2: Reservar memoria en Device.
	unsigned char *d_img, *d_out;
	cudaMalloc((void**) &d_img, bytes);
	cudaMalloc((void**) &d_out, bytes);

	//Paso 3: Transferir los datos de Host to Device.
	cudaMemcpy(d_img, h_img, bytes, cudaMemcpyHostToDevice);

	//Paso 4.1: Lanzar la funcion serial de la CPU.
	tic = cpuTime();
	contrasteCPU(h_img, h_out, factor, alto, ancho);
	toc = cpuTime();
	tictocS = toc - tic;

	//Paso 4.2: Lanzar la funcion paralela de la GPU.
	dim3 hilos(16, 16);
	dim3 bloques((ancho + hilos.x - 1) / hilos.x, (alto + hilos.y - 1) / hilos.y);

	contrasteGPU<<<bloques, hilos>>>(d_img, d_out, factor, alto, ancho); //calentamiento
	cudaDeviceSynchronize();

	tic = cpuTime();
	contrasteGPU<<<bloques, hilos>>>(d_img, d_out, factor, alto, ancho);
	cudaDeviceSynchronize();
	toc = cpuTime();
	tictocP = toc - tic;

	//Paso 5: Transferir los resultados a la CPU.
	cudaMemcpy(h_res, d_out, bytes, cudaMemcpyDeviceToHost);

	// Mostrar resultados (solo una esquina por el tamano de la imagen).
	printf("Factor de contraste: %f\n\n", factor);
	mostrarImagen("Imagen original", h_img, alto, ancho, 4);
	mostrarImagen("Resultado CPU", h_out, alto, ancho, 4);
	mostrarImagen("Resultado GPU", h_res, alto, ancho, 4);
	printf("\n");

	// Verificamos resultados de la GPU y calculamos SpeedUp
	checkResult(h_out, h_res, alto, ancho);
	printf("Tiempo de ejecucion en CPU: %f\n", tictocS);
	printf("Tiempo de ejecucion en GPU: %f\n", tictocP);
	printf("SpeedUp: %lf\n", tictocS/tictocP);

	//Paso 6: Liberar memoria de ambos lados.
	free(h_img);
	free(h_out);
	free(h_res);
	cudaFree(d_img);
	cudaFree(d_out);

	return 0;
}

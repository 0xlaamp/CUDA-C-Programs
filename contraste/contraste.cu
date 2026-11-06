/* 
	Programa : Aumento de Contraste Lineal de una imagen digital utilizando CUDA.

	Lee una imagen real (JPG, PNG, BMP, ...) y multiplica el valor de cada
	pixel por un factor de contraste constante. Los valores se mantienen en el
	rango valido [0, 255] (si el resultado excede 255 se satura a 255).

		salida[i] = min(255, entrada[i] * factor)

	Una imagen a color RGB es una matriz 2D donde cada pixel tiene 3 canales
	(rojo, verde, azul) y cada canal es un unsigned char (0 a 255). Se almacena
	linealmente en memoria (row-major, canales intercalados: R G B R G B ...),
	por lo que el canal c del pixel (fila, col) esta en la posicion:

		(fila * ancho + col) * 3 + c

	Se utiliza un hilo por pixel con coordenadas 2D, y cada hilo procesa los 3
	canales de su pixel.

	Elementos:

	imagen  ->  Imagen de entrada (se carga siempre como RGB, 3 canales).
	factor  ->  Factor de contraste (> 1 aumenta, < 1 disminuye).
	salida  ->  Imagen resultado (se guarda una version de CPU y otra de GPU).

	Librerias externas (de un solo archivo .h, dominio publico):

	stb_image.h        -> lectura de imagenes.
	stb_image_write.h  -> escritura de imagenes.
	Descarga: https://github.com/nothings/stb  (colocar ambos .h junto a este archivo)

	Compilacion:   nvcc contraste.cu -o contraste
	Uso:           ./contraste foto.jpg 1.5
	               (argumentos opcionales: imagen.jpg y factor 1.5 por defecto)

	Ejemplo:

	Entrada: pixel = (10, 100, 200), factor = 1.5
	Salida:  pixel = (15, 150, 255)    (200 * 1.5 = 300 -> se satura a 255)

	El programa incluye una version en CPU y una version para CUDA

*/

#include <stdio.h>
#include <sys/time.h>
#include <stdbool.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define CANALES 3

/* funcion que calcula el tiempo que tarda una CPU */

double cpuTime(){
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-06);
}

/* funcion para validar el resultado */

void checkResult(unsigned char *h_ref, unsigned char *h_res, int alto, int ancho){
	bool match = true;
	for(int i = 0; i < alto * ancho * CANALES; i++){
		if(h_ref[i] != h_res[i]){
			int pixel = i / CANALES;
			printf("Error: valor distinto en pixel (%d, %d) canal %d (CPU: %d, GPU: %d)\n",
					pixel / ancho, pixel % ancho, i % CANALES, h_ref[i], h_res[i]);
			match = false;
			break;
		}
	}
	if(match){
		printf("Resultado correcto\n");
	}
}

/* funcion que muestra en pantalla los primeros pixeles (R,G,B) de la imagen */

void mostrarPixeles(const char *nombre, unsigned char *img, int n){
	printf("%s: [ ", nombre);
	for(int i = 0; i < n; i++){
		printf("(%d,%d,%d) ", img[i*CANALES], img[i*CANALES+1], img[i*CANALES+2]);
	}
	printf("... ]\n");
}

/* funcion aumento de contraste para CPU (secuencial) 
 *
 * Obs: el producto se calcula en float y se satura a 255 antes de regresar
 * 		a unsigned char, de lo contrario el valor daria la vuelta (overflow).
 */

void contrasteCPU(unsigned char *entrada, unsigned char *salida, float factor, int alto, int ancho){
	for(int i = 0; i < alto; i++){
		for(int j = 0; j < ancho; j++){
			int pos = (i * ancho + j) * CANALES;
			for(int c = 0; c < CANALES; c++){
				float v = entrada[pos + c] * factor;
				if(v > 255.0f) v = 255.0f;
				if(v < 0.0f)   v = 0.0f;
				salida[pos + c] = (unsigned char)(v + 0.5f);   //redondeo al entero mas cercano
			}
		}
	}
}

/* funcion aumento de contraste para GPU (coordenadas 2D, un hilo por pixel) */

__global__ void contrasteGPU(unsigned char *entrada, unsigned char *salida, float factor, int alto, int ancho){
	int col  = blockDim.x * blockIdx.x + threadIdx.x;
	int fila = blockDim.y * blockIdx.y + threadIdx.y;

	if(fila < alto && col < ancho){
		int pos = (fila * ancho + col) * CANALES;
		for(int c = 0; c < CANALES; c++){
			float v = entrada[pos + c] * factor;
			if(v > 255.0f) v = 255.0f;
			if(v < 0.0f)   v = 0.0f;
			salida[pos + c] = (unsigned char)(v + 0.5f);
		}
	}
}


int main(int argc, char **argv){

	//Paso 1: Reservar memoria en Host e inicializar los datos (cargar la imagen).
	double tic, toc, tictocS = 0, tictocP = 0;

	const char *archivo = (argc > 1) ? argv[1] : "imagen.jpg";
	float factor = (argc > 2) ? (float) atof(argv[2]) : 1.5f;

	int ancho, alto, canalesOriginal;
	unsigned char *h_img = stbi_load(archivo, &ancho, &alto, &canalesOriginal, CANALES); // fuerza RGB
	if(h_img == NULL){
		printf("No se pudo abrir la imagen '%s': %s\n", archivo, stbi_failure_reason());
		return 1;
	}
	printf("Imagen: %s (%d x %d pixeles, %d canales originales -> %d canales RGB)\n",
			archivo, ancho, alto, canalesOriginal, CANALES);
	printf("Factor de contraste: %f\n\n", factor);

	size_t bytes = (size_t) ancho * alto * CANALES * sizeof(unsigned char);

	unsigned char *h_out = (unsigned char*) malloc(bytes);   // resultado CPU
	unsigned char *h_res = (unsigned char*) malloc(bytes);   // resultado GPU

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

	// Mostrar resultados (solo los primeros pixeles por el tamano de la imagen).
	mostrarPixeles("Imagen original", h_img, 5);
	mostrarPixeles("Resultado CPU  ", h_out, 5);
	mostrarPixeles("Resultado GPU  ", h_res, 5);
	printf("\n");

	// Guardar las imagenes resultantes (PNG sin perdida, para poder compararlas).
	stbi_write_png("contraste_cpu.png", ancho, alto, CANALES, h_out, ancho * CANALES);
	stbi_write_png("contraste_gpu.png", ancho, alto, CANALES, h_res, ancho * CANALES);
	printf("Imagenes guardadas: contraste_cpu.png, contraste_gpu.png\n\n");

	// Verificamos resultados de la GPU y calculamos SpeedUp
	checkResult(h_out, h_res, alto, ancho);
	printf("Tiempo de ejecucion en CPU: %f\n", tictocS);
	printf("Tiempo de ejecucion en GPU: %f\n", tictocP);
	printf("SpeedUp: %lf\n", tictocS/tictocP);

	//Paso 6: Liberar memoria de ambos lados.
	stbi_image_free(h_img);
	free(h_out);
	free(h_res);
	cudaFree(d_img);
	cudaFree(d_out);

	return 0;
}

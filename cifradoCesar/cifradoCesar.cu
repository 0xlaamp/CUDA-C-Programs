/* 
	Programa : Cifrado de Cesar utilizando CUDA.
	
	El cifrado de cesar es un metodo de criptogradia mas antiguos y sencillos,
	consiste en sustituir cada letra de un mensaje por otra letra que se encuentra
	un número fijo de posiciones mas adelante en el alfabeto.

	Elementos:

	Texto plano   ->  El mensaje original.
	Texto cifrado ->  El resultado despues de cifrar.
	Clave (k)     ->  El desplazamiento, un entero.
	Alfabeto      ->  Conjunto de simbolos sobre el que se opera (26 letras en ingles).

    Ejemplos:

	Entrada: s = "Hola", k = 3
	Salida:  "KROD"

	Entrada: s = "Hello, World!", k = 5
	Salida:  "Mjqqt, Btwqi!"
	
	Notas sobre el algoritmo: 
	- Comas, espacio, numeros y signos se quedan igual.
	- El alfabeto da la vuelta despues de la Z.

	El programa incluye una version en CPU y una version para CUDA

*/

#include <stdio.h>
#include <sys/time.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>

/* funcion que calcula el tiempo que tarda una CPU */

double cpuTime(){
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-06);
}

/* funcion para leer un archivo */

char* leerArchivo(const char *nombre, int *size){
    FILE *f = fopen(nombre, "rb");
    if (f == NULL) return NULL;

    fseek(f, 0, SEEK_END);
    *size = ftell(f);
    rewind(f);

    char *buffer = (char*) malloc(*size + 1);
    fread(buffer, 1, *size, f);
    buffer[*size] = '\0';

    fclose(f);
    return buffer;
}





/* funcion para validar el resultado */

void checkResult(char *h_A, char *h_res_A, int size){
        bool match = true;
        for(int i=0; i<size; i++){
                if(h_A[i] != h_res_A[i]){
                        printf("Error: cifrado distinto en la posicion %d (CPU: '%c', GPU: '%c')\n", i, h_A[i], h_res_A[i]);
                        match = false;
                        break;
                }
        }
        if(match){
                printf("Cifrado correcto\n");
        }

}

/* funcion que muestra en pantalla el texto de un arreglo*/

void mostrarTexto(char *texto, int size){
	printf("[ ");
	for (int i = 0; i < size; i++){
		printf("%c ", texto[i]);
	}
	printf("]\n");
} 



/* funcion cifrado cesar para CPU (secuencial) 
 *
 * Obs: En C un char es en realidad un número entero pequeño, el código ASCII de 
 * 		cada caracter y opera con ese número. (A -> 65, B -> 66, ..., Z -> 90, ...)
 * 
*/

void cifradoCesarCPU(char *texto, char *texto_cifrado, int k, int size){
	k = ((k % 26) + 26) % 26;
	for (int i = 0; i < size ; i++) {
		char c = texto[i];
		if (c >= 'A' && c <= 'Z'){
			texto_cifrado[i] = 'A' + (c + k - 'A') % 26;  //el modulo aqui permite darle la vuelta al abecedario.
		} else if (c >= 'a' && c <= 'z'){
			texto_cifrado[i] = 'a' + (c - 'a' + k) % 26;
		} else {
			texto_cifrado[i] = c;
		}
	}
}

/* funcion cifrado cesar para GPU */

__global__ void cifradoCesarGPU(char *texto, char *texto_cifrado, int k, int size){
	k = ((k % 26) + 26) % 26;
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < size){
		char c = texto[idx];

		if (c >= 'A' && c <= 'Z'){
				texto_cifrado[idx] = 'A' + (c + k - 'A') % 26;  //el modulo aqui permite darle la vuelta al abecedario.
																//
		} else if (c >= 'a' && c <= 'z'){
				texto_cifrado[idx] = 'a' + (c - 'a' + k) % 26;

		} else {
				texto_cifrado[idx] = c;
		}
	}
}



int main(void){

	//Paso 1: Reservar memoria en Host e inicializar los datos.
	double tic, toc, tictocS = 0, tictocP = 0;

	int size;
	char *h_s = leerArchivo("cuento.txt", &size);
	if(h_s == NULL){
		printf("No se pudo abrir el archivo\n");
		return 1;
	}

	char *h_cifrado= (char*) malloc(size + 1);
	char *h_res = (char*) malloc(size +1);
	int k = 11;

	//Paso 2: Reservar memoria en Device.
	char *d_s, *d_cifrado;
	cudaMalloc((void**) &d_s, size);
	cudaMalloc((void**) &d_cifrado, size);

	//Paso 3: Transferir los datos de Host to Device.
	cudaMemcpy(d_s, h_s, size, cudaMemcpyHostToDevice);

	//Paso 4.1: Lanzar la funcion serial de la CPU.
	tic = cpuTime();
	cifradoCesarCPU(h_s, h_cifrado, k, size);
	toc = cpuTime();
	tictocS = toc - tic;

	//Paso 4.2: Lanzar la funcion paralela de la GPU.
	dim3 hilos(32);
 	dim3 bloques((size + hilos.x - 1) / hilos.x);

	cifradoCesarGPU<<<bloques, hilos>>>(d_s, d_cifrado, k, size); //calentamiento
	cudaDeviceSynchronize();
										   
	tic = cpuTime();
	cifradoCesarGPU<<<bloques, hilos>>>(d_s, d_cifrado, k, size);
	cudaDeviceSynchronize();
	toc = cpuTime();
	tictocP = toc - tic;

	//Paso 5: Transferir los resultados a la CPU.
	cudaMemcpy(h_res, d_cifrado, size, cudaMemcpyDeviceToHost);


	
	// Mostrar resultados.
	printf("Texto original: %s\n", h_s);
    printf("Texto cifrado por CPU:  %s\n", h_cifrado);
	printf("Texto cifrado por GPU:  %s\n\n", h_res);

	// Verificamos resultados de la GPU y calculamos SpeedUP

	checkResult(h_cifrado, h_res, size);
	printf("Tiempo de ejecucion en CPU: %f\n", tictocS);
	printf("Tiempo de ejecicion en GPU: %f\n", tictocP);
  	printf("SpeedUp: %lf\n", tictocS/tictocP);
	

	//Paso 6: Liberar memoria de ambos lados.
	
	free(h_s);
	cudaFree(d_s);
	cudaFree(d_cifrado);


	
	return 0;
}

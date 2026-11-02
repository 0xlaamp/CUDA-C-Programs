/* 
  Un programa en CUDA (archivos con extension .cu) necesitan de tres cosas 
  para su ejecucion:
  (1) Una GPU NVIDIA
  (2) El driver de la GGU
  (3) El CUDA Toolkit (nvcc, bibliotecas y herramientas)
  
  Un programa CUDA es en realidad una combinacion entre código de HOST y
  código a ejecutar por DEVICE, el compilador (nvcc) es quien sabe 
  como distinguirlos y separarlos. La parte correspondiente a el HOST se 
  la delega al compilador de C (gcc) y el codigo del DEVICE lo compila el 
  mismo.

  El HOST es quien lleva el flujo principal del programa, cuando llega a 
  una seccion de codigo destinado a ser procesado por el DEVICE (como una 
  llamada a un Kernel) eso el no lo ejecuta, inmediatamente el HOST
  entrega esa seccion del código a la GPU. La GPU ya ejecuta dicho kernel
  por su cuenta. 

  OBS: La CPU no espera, sigue con la linea inmediatamente despues del 
  codigo correspondiente a la GPU, mientras que la GPU se mantiene trabajando.
  Lo recomendable es colocar barreras para detener el flujo de la CPU.

*/

/*

  El flujo de ejecucion tipica de todo programa en CUDA:
  Todo programa en CUDA obedece a un modelo de arquiotectura heterogenea, donde
  HOST y DEVICE se comunican y sincronizan entre si. Existe un estandar
  bajo el cual todo programa en CUDA es diseñado el cual consta de solo 6 pasos:

  (1) Reservar memoria en HOST e inicializar los datos:
      Utiliza Malloc() para reservar memoria en RAM y luego puedes iniciar 
      a la estructura generada. Que por cierto, Malloc(bytes) regresa un apuntador
      el cual es basicamente es la direccion en memoria en la que empieza la
      seccion que reservaste.

  (2) Reservar memoria en DEVICE:
     Utiliza CudaMalloc() para reservar memoria en el DEVICE

  (3) Copia/transfiere los datos de HOST a DEVICE
    Utiliza la funcion cudaMemcpy HostToDevice

  (4) Lanza el kernel con suficientes bloques para cubrir todos los datos:
    Establece una cantidad de hilos fija y calcula a partir de ellos la 
    cantidad de bloques necesarios para cubrir todos los datos.

  (5) Copia/transfiere el resultado del device al host (DeviceToHost)

  (6) Libera la memoria en DEVICE (cudaFree) y en HOST (free)

*/




/* Programa : Suma de Matrices */

#include <stdio.h>
#include <sys/time.h>

#define N 50000000


/* funcion para inicializar el arreglo de flaots */

void initialData(float *A, int size){
  
  srand(0);
  float min = 1.0f;
  float max = 100.0f;


  for(int i=0; i < size; i++){
    A[i] = min + ((float)rand() / (float)RAND_MAX * (max - min));
  }
}


/* funcion que muestra el vector resultado */

void showVector(float *A, int size){
  for(int i=0 ; i < size; i++){
    printf("V[%d] = %f \n",i, A[i]);
  }
}


/* funcion que suma dos arreglos en CPU (algoritmo SERIAL) */

void sumaOnCPU(float *h_A, float *h_B, float *h_C, int size){
  for(int i=0 ; i < size ; i++){
    h_C[i] = h_A[i] + h_B[i];
  }

}

/* funcion que calcula el tiempo que tarda una CPU */

double cpuTime(){
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-06);
}


/* funcion que suma dos arreglos en GPU (algoritmo paralelo) */

__global__ void sumaOnGPU(float *d_A, float *d_B, float *d_C, int size){

  int hilo_ID = threadIdx.x + blockIdx.x * blockDim.x;
  if(hilo_ID < size){
    d_C[hilo_ID] = d_A[hilo_ID] + d_B[hilo_ID];
  }
}


/* funcion para validar el resultado */

void checkResult(float *h_A, float *h_res_A, int size){
	double epsilon = 1.0E-8;
	bool match = 1;
	for(int i=0; i<size; i++){
		if(abs(h_A[i] - h_res_A[i]) > epsilon){
			printf("Error: suma distinta\n");
			match = 0;
			break;
		}
	}
	if(match){
		printf("Suma correcta\n");
	}

}



int main(int argc, char* argv[]){
  //Variables para medir el tiempo

  double tic, toc, tictocS = 0, tictocP = 0, tictocDtoH = 0, tictocHtoD = 0, tictocDMalloc = 0, tictocHMalloc = 0;


  // Paso 1: Reservar memoria en HOST e inicializar los datos.

  printf("El tamaño de los arreglos es: %d\n", N);
  size_t nBytes = N * sizeof(float);
  float *h_A, *h_B, *h_C, *h_res;
  tic = cpuTime();
  h_A = (float *) malloc(nBytes);
  h_B = (float *) malloc(nBytes);
  h_C = (float *) malloc(nBytes);
  toc = cpuTime();
  tictocHMalloc = toc - tic;
  h_res = (float *) malloc(nBytes);

  initialData(h_A, N);
  initialData(h_B, N);

  memset(h_C,0,nBytes);
	memset(h_res,0,nBytes);

  // Paso 2: Reservar memoria en DEVICE.

  tic = cpuTime();

  float *d_A, *d_B, *d_C;
  cudaMalloc((void**) &d_A, nBytes);
  cudaMalloc((void**) &d_B, nBytes);
  cudaMalloc((void**) &d_C, nBytes);

  toc = cpuTime();
  tictocDMalloc = toc - tic;

  // Paso 3: Transferir los datos de HOST a DEVICE.

  tic = cpuTime();
  cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, h_C, nBytes, cudaMemcpyHostToDevice);
  toc = cpuTime();
  tictocHtoD = toc - tic;

  // Paso 4.1: Lanzar la funcion serial de CPU.

  tic = cpuTime();
  sumaOnCPU(h_A, h_B, h_C, N);
  toc = cpuTime();
  tictocS = toc - tic;
  printf("El tiempo de ejecucion serial (CPU): %lf\n", tictocS);

  // Paso 4.2: Lanzar el kernel para la GPU.

  dim3 hilos(256);
  dim3 bloques((N + hilos.x - 1) / hilos.x); // Esta es la formula para cubrir todos los datos.
  
  //dim3 bloques(1);
  //dim3 hilos(1);

  sumaOnGPU<<<bloques,hilos>>>(d_A,d_B,d_C,N); //calentamiento
  cudaDeviceSynchronize();

  printf("Cantidad de Bloques: (%u, %u, %u)\n", bloques.x, bloques.y, bloques.z);
  printf("Cantidad de Hilos: (%u, %u, %u)\n", hilos.x, hilos.y, hilos.z);
  
  tic = cpuTime();
  sumaOnGPU<<<bloques,hilos>>>(d_A,d_B,d_C,N);

  //sumaOnGPU<<< bloques,hilos >>>(d_A, d_B, d_C, N);

  cudaDeviceSynchronize();
  toc = cpuTime();
  tictocP = toc - tic;
  printf("El tiempo de ejecucion paralela (GPU): %lf\n", tictocP);


  // Paso 5: Transferir los datos de DEVICTO a HOST.
  tic = cpuTime();
  cudaMemcpy(h_res, d_C, nBytes, cudaMemcpyDeviceToHost);
  toc = cpuTime();
  tictocDtoH = toc - tic;

  // Verificamos resultados de la GPU y calculamos SpeedUP
  printf("SpeedUp: %lf\n", tictocS/tictocP);
  checkResult(h_res,h_C,N);
  // showVector(h_C, N);   // -> Imprime el vector h_C 

  printf("El tiempo de reservar memoria en Host (CPU): %f\n", tictocHMalloc);
  printf("El tiempo de reservar memoria en Devide (GPU): %f\n", tictocDMalloc);
  printf("El tiempo de trasnferir los datos a Device (GPU): %f\n", tictocHtoD);
  printf("El tiempo de copiar los datos de Device to Host (GPU -> CPU): %f\n", tictocDtoH);


  // Paso 6: Liberamos memoria de Host y Device.
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_res);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  
  return 0;

}

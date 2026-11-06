# CUDA C Programs

Colección de programas variados escritos en **CUDA C/C++**.
Este repositorio reúne prácticas, ejercicios y algoritmos variados que resuelven distintos
problemas aprovechando el cómputo paralelo en GPU.

## Sobre este repositorio

Aquí encontrarás, por ejemplo:

- Prácticas utilizando CUDA (kernels, hilos, bloques, memoria global y compartida, etc.)
- Implementaciones paralelas de algoritmos clásicos
- Pruebas y experimentos de rendimiento (CPU vs. GPU)
- Programas variados que resuelven problemas diversos

> **Nota:** Los proyectos más grandes y complejos de CUDA viven en repositorios independientes, 
donde cada repositorio abarca un único proyecto. Este repositorio esta pensado solo para programas 
pequeños y variados entre si.

## Estructura

Cada programa se guarda en su propia carpeta (o archivo `.cu`), con un nombre que describe lo que hace:

```
CUDA-C-Programs/
├── sumaVectores/
│   └── vector_add.cu
├── sumaMatrices/
│   └── matmul.cu
├── ...
└── README.md
```

## Requisitos

- GPU NVIDIA compatible con CUDA
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) instalado (incluye `nvcc`)
- Un compilador de C/C++ compatible (por ejemplo, `gcc` en Linux o MSVC en Windows)

Para verificar la instalación:

```bash
nvcc --version
nvidia-smi
```

## Compilación y ejecución

Para compilar cualquier programa:

```bash
nvcc programa.cu -o programa
```

Para ejecutarlo:

```bash
./programa
```

En Windows, el ejecutable será `programa.exe`.

## Lista de programas

| Programa | Descripción |
|----------|-------------|
| sumaVectores  | Introduccion al modelo de programacion en CUDA paralelizando la suma de vectores |
| sumaMatrices  | Paralelizacion de la suma matricial en memoria contigua utilizando CUDA |
| cifradoCesar  | Paralelizacion de el algortimo cifrado cesar utilizando CUDA |
| fma           | Calcula D[i] = A[i] * B[i] + C[i] con 2²⁴ elementos y un hilo por elemento |
| senoCoseno    | Calcula sinf y cosf cobre angulos aleatorios entre 0 y 2π. |
| escalarMatriz | Multiplica una matriz por un escalar usando coordenadas 2D. |
| contraste     | Multiplica los valores de los pixeles de una imagen por un factor de contraste constante. |


## Contribuciones

Este es un repositorio personal de aprendizaje, pero las sugerencias, correcciones u optimizaciones son bienvenidas. Si encuentras algo que se pueda mejorar, abre un *issue* o un *pull request*.

@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvarsall.bat" x64

set NVCC_FLAGS=-allow-unsupported-compiler -Wno-deprecated-gpu-targets -Xcompiler "/utf-8"

echo Compiling week1...
cd week1\day1
nvcc %NVCC_FLAGS% -o device_query.exe device_query.cu
cd ..\day2
nvcc %NVCC_FLAGS% -o hello_world.exe hello_world.cu
nvcc %NVCC_FLAGS% -o hello_advanced.exe hello_advanced.cu
cd ..\day3
nvcc %NVCC_FLAGS% -o thread_index.exe thread_index.cu
nvcc %NVCC_FLAGS% -o array_index.exe array_index.cu
cd ..\day4
nvcc %NVCC_FLAGS% -o vector_add.exe vector_add.cu
nvcc %NVCC_FLAGS% -o vector_add_benchmark.exe vector_add_benchmark.cu
cd ..\day5
nvcc %NVCC_FLAGS% -o memory_basics.exe memory_basics.cu
nvcc %NVCC_FLAGS% -o unified_memory.exe unified_memory.cu
cd ..\day6
nvcc %NVCC_FLAGS% -o 2d_indexing.exe 2d_indexing.cu
nvcc %NVCC_FLAGS% -o stride_pattern.exe stride_pattern.cu
cd ..\day7
nvcc %NVCC_FLAGS% -o ex1_square.exe ex1_square.cu
nvcc %NVCC_FLAGS% -o ex2_dot_product.exe ex2_dot_product.cu
nvcc %NVCC_FLAGS% -o ex3_matrix_scale.exe ex3_matrix_scale.cu
nvcc %NVCC_FLAGS% -o ex4_find_max.exe ex4_find_max.cu

echo Compiling week2...
cd ..\..\week2\day8
nvcc %NVCC_FLAGS% -o warp_info.exe warp_info.cu
nvcc %NVCC_FLAGS% -o divergence_demo.exe divergence_demo.cu
cd ..\day9
nvcc %NVCC_FLAGS% -o vector_add_optimized.exe vector_add_optimized.cu
cd ..\day10
nvcc %NVCC_FLAGS% -o matrix_mul_basic.exe matrix_mul_basic.cu
cd ..\day11
nvcc %NVCC_FLAGS% -o matrix_mul_tiled.exe matrix_mul_tiled.cu
cd ..\day12
nvcc %NVCC_FLAGS% -o atomic_ops.exe atomic_ops.cu
cd ..\day13
nvcc %NVCC_FLAGS% -o error_handling.exe error_handling.cu
cd ..\day14
nvcc %NVCC_FLAGS% -o matrix_mul_complete.exe matrix_mul_complete.cu

echo Compiling week3...
cd ..\..\week3\day15
nvcc %NVCC_FLAGS% -o memory_coalescing.exe memory_coalescing.cu
cd ..\day16
nvcc %NVCC_FLAGS% -o constant_memory.exe constant_memory.cu
cd ..\day17
nvcc %NVCC_FLAGS% -o scan_basic.exe scan_basic.cu
cd ..\day18
nvcc %NVCC_FLAGS% -o reduction_basic.exe reduction_basic.cu
cd ..\day19
nvcc %NVCC_FLAGS% -o histogram.exe histogram.cu
cd ..\day20
nvcc %NVCC_FLAGS% -o bitonic_sort.exe bitonic_sort.cu
cd ..\day21
nvcc %NVCC_FLAGS% -o image_processing.exe image_processing.cu

echo Compiling week4...
cd ..\..\week4\day22
nvcc %NVCC_FLAGS% -o image_filters.exe image_filters.cu
cd ..\day23
nvcc %NVCC_FLAGS% -o streams.exe streams.cu
cd ..\day24
nvcc %NVCC_FLAGS% -o multi_gpu.exe multi_gpu.cu
cd ..\day26
nvcc %NVCC_FLAGS% -o unified_memory_advanced.exe unified_memory_advanced.cu

echo Done!

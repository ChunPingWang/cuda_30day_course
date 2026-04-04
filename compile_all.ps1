# PowerShell script to compile all CUDA programs

# Import Visual Studio module
$vsPath = "C:\Program Files\Microsoft Visual Studio\18\Insiders"
$vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"

# Run compilation through cmd with VS environment
$batch = @"
@echo off
call "$vcvars"
set NVCC_FLAGS=-allow-unsupported-compiler -Wno-deprecated-gpu-targets -Xcompiler /utf-8

cd /d "C:\Users\Rex Wang\workspace\cuda\cuda_30day_course"

echo Compiling week1/day1...
nvcc %NVCC_FLAGS% -o week1\day1\device_query.exe week1\day1\device_query.cu

echo Compiling week1/day2...
nvcc %NVCC_FLAGS% -o week1\day2\hello_world.exe week1\day2\hello_world.cu
nvcc %NVCC_FLAGS% -o week1\day2\hello_advanced.exe week1\day2\hello_advanced.cu

echo Compiling week1/day3...
nvcc %NVCC_FLAGS% -o week1\day3\thread_index.exe week1\day3\thread_index.cu
nvcc %NVCC_FLAGS% -o week1\day3\array_index.exe week1\day3\array_index.cu

echo Compiling week1/day4...
nvcc %NVCC_FLAGS% -o week1\day4\vector_add.exe week1\day4\vector_add.cu
nvcc %NVCC_FLAGS% -o week1\day4\vector_add_benchmark.exe week1\day4\vector_add_benchmark.cu

echo Compiling week1/day5...
nvcc %NVCC_FLAGS% -o week1\day5\memory_basics.exe week1\day5\memory_basics.cu
nvcc %NVCC_FLAGS% -o week1\day5\unified_memory.exe week1\day5\unified_memory.cu

echo Compiling week1/day6...
nvcc %NVCC_FLAGS% -o week1\day6\2d_indexing.exe week1\day6\2d_indexing.cu
nvcc %NVCC_FLAGS% -o week1\day6\stride_pattern.exe week1\day6\stride_pattern.cu

echo Compiling week1/day7...
nvcc %NVCC_FLAGS% -o week1\day7\ex1_square.exe week1\day7\ex1_square.cu
nvcc %NVCC_FLAGS% -o week1\day7\ex2_dot_product.exe week1\day7\ex2_dot_product.cu
nvcc %NVCC_FLAGS% -o week1\day7\ex3_matrix_scale.exe week1\day7\ex3_matrix_scale.cu
nvcc %NVCC_FLAGS% -o week1\day7\ex4_find_max.exe week1\day7\ex4_find_max.cu

echo Compiling week2/day8...
nvcc %NVCC_FLAGS% -o week2\day8\warp_info.exe week2\day8\warp_info.cu
nvcc %NVCC_FLAGS% -o week2\day8\divergence_demo.exe week2\day8\divergence_demo.cu

echo Compiling week2/day9...
nvcc %NVCC_FLAGS% -o week2\day9\vector_add_optimized.exe week2\day9\vector_add_optimized.cu

echo Compiling week2/day10...
nvcc %NVCC_FLAGS% -o week2\day10\matrix_mul_basic.exe week2\day10\matrix_mul_basic.cu

echo Compiling week2/day11...
nvcc %NVCC_FLAGS% -o week2\day11\matrix_mul_tiled.exe week2\day11\matrix_mul_tiled.cu

echo Compiling week2/day12...
nvcc %NVCC_FLAGS% -o week2\day12\atomic_ops.exe week2\day12\atomic_ops.cu

echo Compiling week2/day13...
nvcc %NVCC_FLAGS% -o week2\day13\error_handling.exe week2\day13\error_handling.cu

echo Compiling week2/day14...
nvcc %NVCC_FLAGS% -o week2\day14\matrix_mul_complete.exe week2\day14\matrix_mul_complete.cu

echo Compiling week3/day15...
nvcc %NVCC_FLAGS% -o week3\day15\memory_coalescing.exe week3\day15\memory_coalescing.cu

echo Compiling week3/day16...
nvcc %NVCC_FLAGS% -o week3\day16\constant_memory.exe week3\day16\constant_memory.cu

echo Compiling week3/day17...
nvcc %NVCC_FLAGS% -o week3\day17\scan_basic.exe week3\day17\scan_basic.cu

echo Compiling week3/day18...
nvcc %NVCC_FLAGS% -o week3\day18\reduction_basic.exe week3\day18\reduction_basic.cu

echo Compiling week3/day19...
nvcc %NVCC_FLAGS% -o week3\day19\histogram.exe week3\day19\histogram.cu

echo Compiling week3/day20...
nvcc %NVCC_FLAGS% -o week3\day20\bitonic_sort.exe week3\day20\bitonic_sort.cu

echo Compiling week3/day21...
nvcc %NVCC_FLAGS% -o week3\day21\image_processing.exe week3\day21\image_processing.cu

echo Compiling week4/day22...
nvcc %NVCC_FLAGS% -o week4\day22\image_filters.exe week4\day22\image_filters.cu

echo Compiling week4/day23...
nvcc %NVCC_FLAGS% -o week4\day23\streams.exe week4\day23\streams.cu

echo Compiling week4/day24...
nvcc %NVCC_FLAGS% -o week4\day24\multi_gpu.exe week4\day24\multi_gpu.cu

echo Compiling week4/day26...
nvcc %NVCC_FLAGS% -o week4\day26\unified_memory_advanced.exe week4\day26\unified_memory_advanced.cu

echo All done!
"@

$batch | Out-File -FilePath "C:\Users\Rex Wang\workspace\cuda\cuda_30day_course\temp_compile.bat" -Encoding ASCII

# Run using cmd
cmd /c "C:\Users\Rex Wang\workspace\cuda\cuda_30day_course\temp_compile.bat"

NVCC = nvcc
NVCC_FLAGS = -Wno-deprecated-gpu-targets

# All source files grouped by week
WEEK1_SRCS = week1/day1/device_query.cu \
             week1/day2/hello_world.cu week1/day2/hello_advanced.cu \
             week1/day3/thread_index.cu week1/day3/array_index.cu \
             week1/day4/vector_add.cu week1/day4/vector_add_benchmark.cu \
             week1/day5/memory_basics.cu week1/day5/unified_memory.cu \
             week1/day6/2d_indexing.cu week1/day6/stride_pattern.cu \
             week1/day7/ex1_square.cu week1/day7/ex2_dot_product.cu \
             week1/day7/ex3_matrix_scale.cu week1/day7/ex4_find_max.cu

WEEK2_SRCS = week2/day8/warp_info.cu week2/day8/divergence_demo.cu \
             week2/day9/vector_add_optimized.cu \
             week2/day10/matrix_mul_basic.cu \
             week2/day11/matrix_mul_tiled.cu \
             week2/day12/atomic_ops.cu \
             week2/day13/error_handling.cu \
             week2/day14/matrix_mul_complete.cu

WEEK3_SRCS = week3/day15/memory_coalescing.cu \
             week3/day16/constant_memory.cu \
             week3/day17/scan_basic.cu \
             week3/day18/reduction_basic.cu \
             week3/day19/histogram.cu \
             week3/day20/bitonic_sort.cu \
             week3/day21/image_processing.cu

WEEK4_SRCS = week4/day22/image_filters.cu \
             week4/day23/streams.cu \
             week4/day24/multi_gpu.cu \
             week4/day26/unified_memory_advanced.cu

ALL_SRCS = $(WEEK1_SRCS) $(WEEK2_SRCS) $(WEEK3_SRCS) $(WEEK4_SRCS)
ALL_BINS = $(ALL_SRCS:.cu=)

.PHONY: all clean week1 week2 week3 week4

all: $(ALL_BINS)

week1: $(WEEK1_SRCS:.cu=)
week2: $(WEEK2_SRCS:.cu=)
week3: $(WEEK3_SRCS:.cu=)
week4: $(WEEK4_SRCS:.cu=)

%: %.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f $(ALL_BINS)

## Matrix Multiplication Algorithms

This program implements various matrix multiplication algorithms including Naive (ijk), Strassen's, and Divide and Conquer methods. Additionally, it provides benchmarks to compare the performance of these algorithms.

### How to Run

To compile and run the program, follow these steps:

1. **Clone the Repository:**

```bash
git clone https://github.com/xhoantran/COP4520.git
cd COP4520
```

2. Install the `benchmark` library:

```bash
git clone https://github.com/google/benchmark.git
cd benchmark
cmake -E make_directory "build"
cmake -E chdir "build" cmake -DBENCHMARK_DOWNLOAD_DEPENDENCIES=on -DCMAKE_BUILD_TYPE=Release ../
```

3. **Compile the Program:**

```bash
g++ main.cpp -std=c++11 -isystem benchmark/include -Lbenchmark/build/src -lbenchmark -lpthread -o main.out 
```

4. **Run the Program:**

```bash
./main.out
```

### Benchmarking

The program includes benchmarks for each matrix multiplication algorithm. Benchmarks are executed with varying matrix sizes to analyze the performance under different complexities.

### Algorithms Implemented

1. **Naive Matrix Multiplication**

   - Algorithm: Implements the standard triple-nested loop approach for matrix multiplication.

2. **Strassen's Matrix Multiplication**

   - Algorithm: A recursive algorithm that divides matrices into submatrices and performs matrix multiplications using a set of seven recursive calls. Time complexity O(n^2.81) is lower than the naive approach (O(n^3))

3. **Strassen's Matrix Multiplication with Multithreading**

   - Algorithm: Utilizes multithreading to compute intermediate matrices concurrently in Strassen's algorithm, aiming to enhance performance.
   - Same time complexity as Strassen's algorithm.

4. **Divide and Conquer Matrix Multiplication**

   - Algorithm: Another recursive approach that divides matrices into submatrices and recursively computes submatrix multiplications.

5. **Divide and Conquer Matrix Multiplication with Multithreading**
   - Algorithm: Utilizes multithreading to compute submatrix multiplications concurrently in the divide and conquer algorithm.

### Notes

- The program utilizes the `benchmark` library for performance measurement and comparison.
- For Strassen's and Divide and Conquer algorithms, multithreading is employed to enhance performance for large matrices.
- Adjust the matrix size range in the benchmarks according to your system's capabilities and requirements.
- Performance may vary depending on hardware and compiler optimizations.

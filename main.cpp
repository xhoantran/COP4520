#include <iostream>
#include <vector>
#include <random>
#include <thread>
#include <benchmark/benchmark.h>

using namespace std;
using namespace std::chrono;

// Naive matrix multiplication
vector<vector<int>> naiveMatrixMult(const vector<vector<int>> &A, const vector<vector<int>> &B)
{
  int n = A.size();
  vector<vector<int>> C(n, vector<int>(n, 0));

  for (int i = 0; i < n; ++i)
  {
    for (int j = 0; j < n; ++j)
    {
      for (int k = 0; k < n; ++k)
      {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }

  return C;
}

// Naive matrix multiplication with multi-threading
vector<vector<int>> naiveMatrixMultMultiThreading(const vector<vector<int>> &A, const vector<vector<int>> &B)
{
  int n = A.size();
  vector<vector<int>> C(n, vector<int>(n, 0));

  // Create threads
  vector<thread> threads;

  for (int i = 0; i < n; ++i)
  {
    threads.emplace_back([&, i]()
                         {
                          for (int j = 0; j < n; ++j)
                          {
                            for (int k = 0; k < n; ++k)
                            {
                              C[i][j] += A[i][k] * B[k][j];
                            }
                          } });
  }

  // Join threads
  for (auto &thread : threads)
  {
    thread.join();
  }

  return C;
}

// Helper function to add two matrices
vector<vector<int>> matrixAddition(const vector<vector<int>> &A, const vector<vector<int>> &B)
{
  int n = A.size();
  vector<vector<int>> C(n, vector<int>(n, 0));

  for (int i = 0; i < n; ++i)
  {
    for (int j = 0; j < n; ++j)
    {
      C[i][j] = A[i][j] + B[i][j];
    }
  }

  return C;
}

// Helper function to subtract two matrices
vector<vector<int>> matrixSubtraction(const vector<vector<int>> &A, const vector<vector<int>> &B)
{
  int n = A.size();
  vector<vector<int>> C(n, vector<int>(n, 0));

  for (int i = 0; i < n; ++i)
  {
    for (int j = 0; j < n; ++j)
    {
      C[i][j] = A[i][j] - B[i][j];
    }
  }

  return C;
}

// Strassen's matrix multiplication
vector<vector<int>> strassenMatrixMult(const vector<vector<int>> &A, const vector<vector<int>> &B)
{
  int n = A.size();
  vector<vector<int>> C(n, vector<int>(n, 0));

  // Base case
  if (n <= 16)
  {
    C = naiveMatrixMult(A, B);
  }
  else
  {
    int halfSize = n / 2;

    // Split matrices into submatrices
    vector<vector<int>> A11(halfSize, vector<int>(halfSize));
    vector<vector<int>> A12(halfSize, vector<int>(halfSize));
    vector<vector<int>> A21(halfSize, vector<int>(halfSize));
    vector<vector<int>> A22(halfSize, vector<int>(halfSize));

    vector<vector<int>> B11(halfSize, vector<int>(halfSize));
    vector<vector<int>> B12(halfSize, vector<int>(halfSize));
    vector<vector<int>> B21(halfSize, vector<int>(halfSize));
    vector<vector<int>> B22(halfSize, vector<int>(halfSize));

    for (int i = 0; i < halfSize; ++i)
    {
      for (int j = 0; j < halfSize; ++j)
      {
        A11[i][j] = A[i][j];
        A12[i][j] = A[i][j + halfSize];
        A21[i][j] = A[i + halfSize][j];
        A22[i][j] = A[i + halfSize][j + halfSize];

        B11[i][j] = B[i][j];
        B12[i][j] = B[i][j + halfSize];
        B21[i][j] = B[i + halfSize][j];
        B22[i][j] = B[i + halfSize][j + halfSize];
      }
    }

    // Calculate intermediate matrices
    vector<vector<int>> M1 = strassenMatrixMult(A11, matrixSubtraction(B12, B22));
    vector<vector<int>> M2 = strassenMatrixMult(matrixAddition(A11, A12), B22);
    vector<vector<int>> M3 = strassenMatrixMult(matrixAddition(A21, A22), B11);
    vector<vector<int>> M4 = strassenMatrixMult(A22, matrixSubtraction(B21, B11));
    vector<vector<int>> M5 = strassenMatrixMult(matrixAddition(A11, A22), matrixAddition(B11, B22));
    vector<vector<int>> M6 = strassenMatrixMult(matrixSubtraction(A12, A22), matrixAddition(B21, B22));
    vector<vector<int>> M7 = strassenMatrixMult(matrixSubtraction(A11, A21), matrixAddition(B11, B12));

    // Calculate resulting submatrices
    vector<vector<int>> C11 = matrixAddition(matrixSubtraction(matrixAddition(M5, M4), M2), M6);
    vector<vector<int>> C12 = matrixAddition(M1, M2);
    vector<vector<int>> C21 = matrixAddition(M3, M4);
    vector<vector<int>> C22 = matrixSubtraction(matrixSubtraction(matrixAddition(M5, M1), M3), M7);

    // Combine resulting submatrices
    for (int i = 0; i < halfSize; ++i)
    {
      for (int j = 0; j < halfSize; ++j)
      {
        C[i][j] = C11[i][j];
        C[i][j + halfSize] = C12[i][j];
        C[i + halfSize][j] = C21[i][j];
        C[i + halfSize][j + halfSize] = C22[i][j];
      }
    }
  }

  return C;
}

// Strassen's matrix multiplication with multi-threading
vector<vector<int>> strassenMatrixMultMultiThreading(const vector<vector<int>> &A, const vector<vector<int>> &B)
{
  int n = A.size();
  vector<vector<int>> C(n, vector<int>(n, 0));

  // Base case
  if (n <= 8)
  {
    C = naiveMatrixMult(A, B);
  }
  else
  {
    int halfSize = n / 2;

    // Split matrices into submatrices
    vector<vector<int>> A11(halfSize, vector<int>(halfSize));
    vector<vector<int>> A12(halfSize, vector<int>(halfSize));
    vector<vector<int>> A21(halfSize, vector<int>(halfSize));
    vector<vector<int>> A22(halfSize, vector<int>(halfSize));

    vector<vector<int>> B11(halfSize, vector<int>(halfSize));
    vector<vector<int>> B12(halfSize, vector<int>(halfSize));
    vector<vector<int>> B21(halfSize, vector<int>(halfSize));
    vector<vector<int>> B22(halfSize, vector<int>(halfSize));

    for (int i = 0; i < halfSize; ++i)
    {
      for (int j = 0; j < halfSize; ++j)
      {
        A11[i][j] = A[i][j];
        A12[i][j] = A[i][j + halfSize];
        A21[i][j] = A[i + halfSize][j];
        A22[i][j] = A[i + halfSize][j + halfSize];

        B11[i][j] = B[i][j];
        B12[i][j] = B[i][j + halfSize];
        B21[i][j] = B[i + halfSize][j];
        B22[i][j] = B[i + halfSize][j + halfSize];
      }
    }

    // Intermediate matrices
    vector<vector<int>> M1, M2, M3, M4, M5, M6, M7;

    // Calculate intermediate matrices concurrently
    vector<thread> threads;
    threads.emplace_back([&]()
                         { M1 = strassenMatrixMult(A11, matrixSubtraction(B12, B22)); });
    threads.emplace_back([&]()
                         { M2 = strassenMatrixMult(matrixAddition(A11, A12), B22); });
    threads.emplace_back([&]()
                         { M3 = strassenMatrixMult(matrixAddition(A21, A22), B11); });
    threads.emplace_back([&]()
                         { M4 = strassenMatrixMult(A22, matrixSubtraction(B21, B11)); });
    threads.emplace_back([&]()
                         { M5 = strassenMatrixMult(matrixAddition(A11, A22), matrixAddition(B11, B22)); });
    threads.emplace_back([&]()
                         { M6 = strassenMatrixMult(matrixSubtraction(A12, A22), matrixAddition(B21, B22)); });
    threads.emplace_back([&]()
                         { M7 = strassenMatrixMult(matrixSubtraction(A11, A21), matrixAddition(B11, B12)); });

    for (auto &thread : threads)
    {
      thread.join();
    }

    // Calculate resulting submatrices
    vector<vector<int>> C11 = matrixAddition(matrixSubtraction(matrixAddition(M5, M4), M2), M6);
    vector<vector<int>> C12 = matrixAddition(M1, M2);
    vector<vector<int>> C21 = matrixAddition(M3, M4);
    vector<vector<int>> C22 = matrixSubtraction(matrixSubtraction(matrixAddition(M5, M1), M3), M7);

    // Combine resulting submatrices
    for (int i = 0; i < halfSize; ++i)
    {
      for (int j = 0; j < halfSize; ++j)
      {
        C[i][j] = C11[i][j];
        C[i][j + halfSize] = C12[i][j];
        C[i + halfSize][j] = C21[i][j];
        C[i + halfSize][j + halfSize] = C22[i][j];
      }
    }
  }

  return C;
}

// Divide and conquer matrix multiplication
vector<vector<int>> divideAndConquerMatrixMult(const vector<vector<int>> &A, const vector<vector<int>> &B)
{
  int n = A.size();
  vector<vector<int>> C(n, vector<int>(n, 0));

  if (n == 1)
  {
    C[0][0] = A[0][0] * B[0][0];
  }
  else
  {
    int halfSize = n / 2;

    // Split matrices into submatrices
    vector<vector<int>> A11(halfSize, vector<int>(halfSize));
    vector<vector<int>> A12(halfSize, vector<int>(halfSize));
    vector<vector<int>> A21(halfSize, vector<int>(halfSize));
    vector<vector<int>> A22(halfSize, vector<int>(halfSize));

    vector<vector<int>> B11(halfSize, vector<int>(halfSize));
    vector<vector<int>> B12(halfSize, vector<int>(halfSize));
    vector<vector<int>> B21(halfSize, vector<int>(halfSize));
    vector<vector<int>> B22(halfSize, vector<int>(halfSize));

    for (int i = 0; i < halfSize; ++i)
    {
      for (int j = 0; j < halfSize; ++j)
      {
        A11[i][j] = A[i][j];
        A12[i][j] = A[i][j + halfSize];
        A21[i][j] = A[i + halfSize][j];
        A22[i][j] = A[i + halfSize][j + halfSize];

        B11[i][j] = B[i][j];
        B12[i][j] = B[i][j + halfSize];
        B21[i][j] = B[i + halfSize][j];
        B22[i][j] = B[i + halfSize][j + halfSize];
      }
    }

    // Recursively compute submatrix multiplications
    vector<vector<int>> C11 = matrixAddition(divideAndConquerMatrixMult(A11, B11),
                                             divideAndConquerMatrixMult(A12, B21));
    vector<vector<int>> C12 = matrixAddition(divideAndConquerMatrixMult(A11, B12),
                                             divideAndConquerMatrixMult(A12, B22));
    vector<vector<int>> C21 = matrixAddition(divideAndConquerMatrixMult(A21, B11),
                                             divideAndConquerMatrixMult(A22, B21));
    vector<vector<int>> C22 = matrixAddition(divideAndConquerMatrixMult(A21, B12),
                                             divideAndConquerMatrixMult(A22, B22));

    // Combine submatrix results
    for (int i = 0; i < halfSize; ++i)
    {
      for (int j = 0; j < halfSize; ++j)
      {
        C[i][j] = C11[i][j];
        C[i][j + halfSize] = C12[i][j];
        C[i + halfSize][j] = C21[i][j];
        C[i + halfSize][j + halfSize] = C22[i][j];
      }
    }
  }

  return C;
}

// Function to divide and conquer matrix multiplication with multi-threading
vector<vector<int>> divideAndConquerMatrixMultMultiThreading(const vector<vector<int>> &A, const vector<vector<int>> &B)
{
  int n = A.size();
  vector<vector<int>> C(n, vector<int>(n, 0));

  if (n <= 8)
  {
    C = naiveMatrixMult(A, B);
  }
  else
  {
    int halfSize = n / 2;

    // Split matrices into submatrices
    vector<vector<int>> A11(halfSize, vector<int>(halfSize));
    vector<vector<int>> A12(halfSize, vector<int>(halfSize));
    vector<vector<int>> A21(halfSize, vector<int>(halfSize));
    vector<vector<int>> A22(halfSize, vector<int>(halfSize));

    vector<vector<int>> B11(halfSize, vector<int>(halfSize));
    vector<vector<int>> B12(halfSize, vector<int>(halfSize));
    vector<vector<int>> B21(halfSize, vector<int>(halfSize));
    vector<vector<int>> B22(halfSize, vector<int>(halfSize));

    for (int i = 0; i < halfSize; ++i)
    {
      for (int j = 0; j < halfSize; ++j)
      {
        A11[i][j] = A[i][j];
        A12[i][j] = A[i][j + halfSize];
        A21[i][j] = A[i + halfSize][j];
        A22[i][j] = A[i + halfSize][j + halfSize];

        B11[i][j] = B[i][j];
        B12[i][j] = B[i][j + halfSize];
        B21[i][j] = B[i + halfSize][j];
        B22[i][j] = B[i + halfSize][j + halfSize];
      }
    }

    // Recursively compute submatrix multiplications using multi-threading
    vector<thread> threads;
    vector<vector<int>> C11, C12, C21, C22;
    threads.emplace_back([&]()
                         { C11 = matrixAddition(divideAndConquerMatrixMult(A11, B11),
                                                divideAndConquerMatrixMult(A12, B21)); });
    threads.emplace_back([&]()
                         { C12 = matrixAddition(divideAndConquerMatrixMult(A11, B12),
                                                divideAndConquerMatrixMult(A12, B22)); });
    threads.emplace_back([&]()
                         { C21 = matrixAddition(divideAndConquerMatrixMult(A21, B11),
                                                divideAndConquerMatrixMult(A22, B21)); });
    threads.emplace_back([&]()
                         { C22 = matrixAddition(divideAndConquerMatrixMult(A21, B12),
                                                divideAndConquerMatrixMult(A22, B22)); });

    // Join threads
    for (auto &thread : threads)
    {
      thread.join();
    }

    // Combine submatrix results
    for (int i = 0; i < halfSize; ++i)
    {
      for (int j = 0; j < halfSize; ++j)
      {
        C[i][j] = C11[i][j];
        C[i][j + halfSize] = C12[i][j];
        C[i + halfSize][j] = C21[i][j];
        C[i + halfSize][j + halfSize] = C22[i][j];
      }
    }
  }

  return C;
}

// Benchmark function for naive matrix multiplication
static void BM_NaiveMatrixMultiplication(benchmark::State &state)
{
  int N = state.range(0);
  vector<vector<int>> A(N, vector<int>(N));
  vector<vector<int>> B(N, vector<int>(N));

  // Fill matrices A and B with random values
  random_device rd;
  mt19937 gen(rd());
  uniform_int_distribution<int> dis(1, 1000);

  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      A[i][j] = dis(gen);
      B[i][j] = dis(gen);
    }
  }

  for (auto _ : state)
  {
    vector<vector<int>> C = naiveMatrixMult(A, B);
  }
  state.SetComplexityN(N);
}

// Benchmark function for naive matrix multiplication with multi-threading
static void BM_NaiveMatrixMultiplicationMultiThreading(benchmark::State &state)
{
  int N = state.range(0);
  vector<vector<int>> A(N, vector<int>(N));
  vector<vector<int>> B(N, vector<int>(N));

  // Fill matrices A and B with random values
  random_device rd;
  mt19937 gen(rd());
  uniform_int_distribution<int> dis(1, 1000);

  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      A[i][j] = dis(gen);
      B[i][j] = dis(gen);
    }
  }

  for (auto _ : state)
  {
    vector<vector<int>> C = naiveMatrixMultMultiThreading(A, B);
  }
  state.SetComplexityN(N);
}

// Benchmark function for Strassen's matrix multiplication
static void BM_StrassenMatrixMultiplication(benchmark::State &state)
{
  int N = state.range(0);
  vector<vector<int>> A(N, vector<int>(N));
  vector<vector<int>> B(N, vector<int>(N));

  // Fill matrices A and B with random values
  random_device rd;
  mt19937 gen(rd());
  uniform_int_distribution<int> dis(1, 1000);

  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      A[i][j] = dis(gen);
      B[i][j] = dis(gen);
    }
  }

  for (auto _ : state)
  {
    vector<vector<int>> C = strassenMatrixMult(A, B);
  }
  state.SetComplexityN(N);
}

// Benchmark function for Strassen's matrix multiplication with multi-threading
static void BM_StrassenMatrixMultiplicationMultiThreading(benchmark::State &state)
{
  int N = state.range(0);
  vector<vector<int>> A(N, vector<int>(N));
  vector<vector<int>> B(N, vector<int>(N));

  // Fill matrices A and B with random values
  random_device rd;
  mt19937 gen(rd());
  uniform_int_distribution<int> dis(1, 1000);

  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      A[i][j] = dis(gen);
      B[i][j] = dis(gen);
    }
  }

  for (auto _ : state)
  {
    vector<vector<int>> C = strassenMatrixMultMultiThreading(A, B);
  }
  state.SetComplexityN(N);
}

// Benchmark function for divide and conquer matrix multiplication
static void BM_DivideAndConquerMatrixMultiplication(benchmark::State &state)
{
  int N = state.range(0);
  vector<vector<int>> A(N, vector<int>(N));
  vector<vector<int>> B(N, vector<int>(N));

  // Fill matrices A and B with random values
  random_device rd;
  mt19937 gen(rd());
  uniform_int_distribution<int> dis(1, 1000);

  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      A[i][j] = dis(gen);
      B[i][j] = dis(gen);
    }
  }

  for (auto _ : state)
  {
    vector<vector<int>> C = divideAndConquerMatrixMult(A, B);
  }
  state.SetComplexityN(N);
}

// Benchmark function for divide and conquer matrix multiplication with multi-threading
static void BM_DivideAndConquerMatrixMultiplicationMultiThreading(benchmark::State &state)
{
  int N = state.range(0);
  vector<vector<int>> A(N, vector<int>(N));
  vector<vector<int>> B(N, vector<int>(N));

  // Fill matrices A and B with random values
  random_device rd;
  mt19937 gen(rd());
  uniform_int_distribution<int> dis(1, 1000);

  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      A[i][j] = dis(gen);
      B[i][j] = dis(gen);
    }
  }

  for (auto _ : state)
  {
    vector<vector<int>> C = divideAndConquerMatrixMultMultiThreading(A, B);
  }
  state.SetComplexityN(N);
}

// Register the benchmarks, 2 <= N <= 512, step size = N * 2, 2, 4, 8, 16, 32, 64, 128, 256, 512
BENCHMARK(BM_NaiveMatrixMultiplication)->RangeMultiplier(2)->Range(2, 512)->Complexity();
BENCHMARK(BM_NaiveMatrixMultiplicationMultiThreading)->RangeMultiplier(2)->Range(2, 512)->Complexity();
BENCHMARK(BM_StrassenMatrixMultiplication)->RangeMultiplier(2)->Range(2, 512)->Complexity();
BENCHMARK(BM_StrassenMatrixMultiplicationMultiThreading)->RangeMultiplier(2)->Range(2, 512)->Complexity();
BENCHMARK(BM_DivideAndConquerMatrixMultiplication)->RangeMultiplier(2)->Range(2, 512)->Complexity();
BENCHMARK(BM_DivideAndConquerMatrixMultiplicationMultiThreading)->RangeMultiplier(2)->Range(2, 512)->Complexity();

// Run the benchmark
BENCHMARK_MAIN();

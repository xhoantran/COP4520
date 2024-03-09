# Threaded Matrix Multiplication in Java

This Java project demonstrates an efficient approach to matrix multiplication using multithreading. The implementation leverages Java's concurrency framework to parallelize the multiplication process, resulting in improved performance for large matrices.

## Features

- Utilizes Java's `Callable` and `Future` to handle concurrent tasks.
- Employs an `ExecutorService` to manage a pool of threads efficiently.
- Offers a scalable solution to matrix multiplication by parallelizing the computation of each row of the resulting matrix.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Java Development Kit (JDK) 8 or higher

### Installing

1. Clone the repository to your local machine:
    ```bash
    git clone https://github.com/xhoantran/COP4502.git
    ```
2. Navigate to the cloned directory.

3. Compile the Java code:
    ```bash
    javac ThreadedMatrixMultiplication.java
    ```

4. Run the compiled program:
    ```bash
    java ThreadedMatrixMultiplication
    ```

## Usage

The main class `ThreadedMatrixMultiplication` contains a `main` method that demonstrates a simple use case of multiplying two predefined matrices. To use this in your project, you can call the `threadedMatrixMultiply` method with two 2D integer arrays (matrices) as parameters:

```java
int[][] A = {
    {1, 2},
    {3, 4}
};

int[][] B = {
    {5, 6},
    {7, 8}
};

int[][] result = ThreadedMatrixMultiplication.threadedMatrixMultiply(A, B);
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- Hat tip to anyone whose code was used
- Inspiration
- etc


import threading
import numpy as np

def multiply_row_by_column(result, A, B, row):
    for i in range(len(B[0])):  # For each column in B
        # Compute the dot product of the row of A and the column of B
        sum = 0
        for j in range(len(A[0])):  # For each element in the row/column
            sum += A[row][j] * B[j][i]
        result[row][i] = sum

# Matrix multiplication function using threads
def threaded_matrix_multiply(A, B):
    num_rows_A = len(A)
    num_cols_B = len(B[0])
    
    # Resultant matrix of zeros
    result = [[0 for _ in range(num_cols_B)] for _ in range(num_rows_A)]
    threads = []

    # Create a new thread for each row of A
    for i in range(num_rows_A):
        thread = threading.Thread(target=multiply_row_by_column, args=(result, A, B, i))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    return result

if __name__ == "__main__":
    A = np.random.randint(10, size=(4, 4))
    B = np.random.randint(10, size=(4, 4))
    result = threaded_matrix_multiply(A, B)
    print("Matrix A:")
    print(A)
    print("Matrix B:")
    print(B)
    print("Result of multiplication:")
    print(np.array(result))

import numpy as np
import multiprocessing as mp
import time

def matrix_multiply(A, B, C, row_start, row_end, col_start, col_end):
    for i in range(row_start, row_end):
        for j in range(col_start, col_end):
            C[i][j] = np.dot(A[i, :], B[:, j])

def parallel_matrix_multiply(A, B, num_threads):
    n = A.shape[0]
    tile_size = n // num_threads
    C = np.zeros((n, n))
    
    processes = []
    for i in range(num_threads):
        for j in range(num_threads):
            row_start = i * tile_size
            row_end = (i + 1) * tile_size
            col_start = j * tile_size
            col_end = (j + 1) * tile_size
            process = mp.Process(target=matrix_multiply, args=(A, B, C, row_start, row_end, col_start, col_end))
            processes.append(process)
            process.start()
    
    for process in processes:
        process.join()
    
    return C

def measure_time(A, B, num_threads):
    start_time = time.time()
    C = parallel_matrix_multiply(A, B, num_threads)
    end_time = time.time()
    return end_time - start_time

if __name__ == '__main__':
    n = 1024  # Matrix size
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)

    thread_counts = [1, 2, 4, 8, 16, 32]
    times = []

    for num_threads in thread_counts:
        time_taken = measure_time(A, B, num_threads)
        times.append((num_threads, time_taken))
        print(f"Threads: {num_threads}, Time taken: {time_taken:.4f} seconds")

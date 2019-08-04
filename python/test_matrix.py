#!/usr/bin/python3
import numpy as np

def test_matrix_mul_notrans():
    array_a = np.arange(0, 5 * 4, 1)
    array_a = array_a + 1
    matrix_a = np.matrix(array_a.reshape(5, 4))
    
    array_b = np.arange(0, 4 * 3, 1)
    array_b = array_b + 1
    matrix_b = np.matrix(array_b.reshape(4, 3))

    matrix_c = np.matmul(matrix_a, matrix_b)
    print ("no transpose no transpose")
    print (matrix_c)

def test_matrix_mul_trans():
    array_a = np.arange(0, 6, 1)
    array_b = np.arange(0, 6, 1)
    matrix_a = np.matrix(array_a.reshape(3, 2))
    matrix_b = np.matrix(array_b.reshape(3, 2))
    matrix_c = np.matmul(matrix_a, matrix_b.T)
    print("no transpose transpose")
    print(matrix_c)

def test_matrix_vector_notrans():
    array_a = np.arange(0, 6, 1)
    array_b = np.arange(0, 2)
    matrix_a = np.matrix(array_a.reshape(3, 2))
    matrix_b = np.matrix(array_b.reshape(2, 1))
    matrix_c = np.matmul(matrix_a, matrix_b);
    print(matrix_c)

def test_matrix_vector_trans():
    array_a = np.arange(0, 6, 1)
    array_b = np.arange(0, 3, 1)
    matrix_a = np.matrix(array_a.reshape(3, 2))
    matrix_b = np.matrix(array_b.reshape(3, 1))
    matrix_c = np.matmul(matrix_a.T, matrix_b);
    print(matrix_c)



if __name__ == "__main__":
    test_matrix_mul_notrans()
    test_matrix_mul_trans()
    test_matrix_vector_notrans()
    test_matrix_vector_trans()

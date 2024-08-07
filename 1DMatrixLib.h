#pragma once
#include <vector>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <functional>
#include <random>

using namespace std;

class Matrix {
public:
    int rows;
    int cols;
    vector<float> data;

    Matrix() : cols(0), rows(0), data({}) {}

    Matrix(int rows, int cols) : cols(cols), rows(rows), data(rows* cols) {}

    void RandInit() {
        std::random_device rd;
        std::uniform_real_distribution<float> dist(-1, 1);
        for (int i = 0; i < rows * cols; i++) {
            data[i] = dist(rd);
        }
    }

    void printMatrix() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                cout << data[i * cols + j] << "\t";
            }
            cout << "\n";
        }
    }

    // Multiplies current object by a scalar
    void multiplyScalar(float n) {
        for (int i = 0; i < rows * cols; i++) {
            data[i] *= n;
        }
    }

    // Adds scalar value to current object
    void addScalar(float n) {
        for (int i = 0; i < rows * cols; i++) {
            data[i] += n;
        }
    }

    // Makes all values in current Matrix object negative
    void negate() {
        for (int i = 0; i < rows * cols; i++) {
            data[i] *= -1;
        }
    }

    // This negates the values in the current object and loads them into another matrix which is returned
    Matrix Negate() {
        Matrix m(rows, cols);
        for (int i = 0; i < rows * cols; i++) {
            m.data[i] = data[i] * -1;
        }
        return m;
    }

    // Current object is manipulated -> other matrix's values are added to the current object only if condition is met
    // otherwise the message will be printed
    void add(Matrix& other) {
        if (rows == other.rows && cols == other.cols) {
            for (int i = 0; i < rows * cols; i++) {
                data[i] += other.data[i];
            }
        }
        else {
            cout << "Dims of matrices must be equal to add both of them. Current object remains UNCHANGED.\n";
        }
    }

    // Static method to represent 'add' method for the whole Matrix class; synthesizes two matrix objects and returns result
    // if condition is not met, Matrix() is returned
    static Matrix add(Matrix& m1, Matrix& m2) {
        Matrix result(m1.rows, m1.cols);
        if (m1.rows == m2.rows && m1.cols == m2.cols) {
            for (int i = 0; i < m1.rows * m1.cols; i++) {
                result.data[i] = m1.data[i] + m2.data[i];
            }
            return result;
        }
        return Matrix();
    }

    // Instance method Add which does same thing as void add method but returns a new matrix object
    Matrix Add(Matrix& other) {
        if (rows == other.rows && cols == other.cols) {
            Matrix output(rows, cols);
            for (int i = 0; i < rows * cols; i++) {
                output.data[i] = data[i] + other.data[i];
            }
            return output;
        }
        return Matrix();
    }

    // Method that subtracts 2 matrices and returns result. It is static in order to represent subtraction method for the whole Matrix class
    static Matrix subtract(Matrix m1, Matrix m2) {
        Matrix result(m1.rows, m1.cols);
        if (m1.rows == m2.rows && m1.cols == m2.cols) {
            for (int i = 0; i < m1.rows * m1.cols; i++) {
                result.data[i] = m1.data[i] - m2.data[i];
            }
            return result;
        }
        return Matrix();
    }

    // Instance method that multiplies other matrix's values with current matrix's values and sets dimensions accordingly using setter methods
    // if condition not met a message is printed
    void multiply(Matrix& other) {
        if (cols != other.rows) {
            cout << "Cols of first matrix should equal rows of second matrix to multiply both of them. Current object remains UNCHANGED.\n";
            return;
        }

        Matrix output(rows, other.cols);
        for (int i = 0; i < output.rows; i++) {
            for (int j = 0; j < output.cols; j++) {
                float sum = 0;
                for (int k = 0; k < cols; k++) {
                    sum += data[i * cols + k] * other.data[k * other.cols + j];
                }
                output.data[i * output.cols + j] = sum;
            }
        }

        *this = output;
    }

    // Static method for the whole class for multiplication; multiplies two matrices and returns result
    // if condition not met Matrix() is returned
    static Matrix multiply(Matrix m1, Matrix m2) {
        if (m1.cols != m2.rows) {
            cout << "Cols of first matrix should equal rows of second matrix to multiply both of them.\n";
            return Matrix();
        }

        Matrix output(m1.rows, m2.cols);
        for (int i = 0; i < output.rows; i++) {
            for (int j = 0; j < output.cols; j++) {
                float sum = 0;
                for (int k = 0; k < m1.cols; k++) {
                    sum += m1.data[i * m1.cols + k] * m2.data[k * m2.cols + j];
                }
                output.data[i * output.cols + j] = sum;
            }
        }

        return output;
    }

    // HADAMARD PRODUCT - ELEMENT WISE MATRIX MULTIPLICATION
    // Multiplies val in 1st matrix to corresponding val in second matrix
    void elementWiseMult(Matrix& other) {
        if (rows == other.rows && cols == other.cols) {
            for (int i = 0; i < rows * cols; i++) {
                data[i] *= other.data[i];
            }
        }
        else {
            cout << "Dims of matrices must be equal to perform element wise multiplication. Current object remains UNCHANGED.\n";
        }
    }

    // Instance method that returns a matrix after doing element-wise multiplication
    // Returns Matrix() if condition not met
    Matrix ElementWiseMult(Matrix& other) {
        if (rows == other.rows && cols == other.cols) {
            Matrix output(rows, cols);
            for (int i = 0; i < rows * cols; i++) {
                output.data[i] = data[i] * other.data[i];
            }
            return output;
        }
        return Matrix();
    }

    // Static method for element-wise multiplication, for the whole Matrix class
    // Returns Matrix() if condition not met
    static Matrix ElementWiseMult(Matrix& m1, Matrix& m2) {
        if (m1.rows == m2.rows && m1.cols == m2.cols) {
            Matrix output(m1.rows, m1.cols);
            for (int i = 0; i < m1.rows * m1.cols; i++) {
                output.data[i] = m1.data[i] * m2.data[i];
            }
            return output;
        }
        return Matrix();
    }

    // Transposes current matrix object and uses setter methods to set dimensions of the modified current object accordingly
    void transpose() {
        Matrix output(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                output.data[j * rows + i] = data[i * cols + j];
            }
        }
        *this = output;
    }

    // Instance method for transpose that returns resultant, transposed matrix
    Matrix Transpose() {
        Matrix output(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                output.data[j * rows + i] = data[i * cols + j];
            }
        }
        return output;
    }

    // Neural Network functions
    static float Sigmoid(float x) {
        return 1.0f / (1 + exp(-x));
    }

    static float dSigmoid(float x) {
        return x * (1 - x);
    }

    void applySigmoid() {
        for (int i = 0; i < rows * cols; i++) {
            data[i] = Sigmoid(data[i]);
        }
    }

    void applySigmoidDerivative() {
        for (int i = 0; i < rows * cols; i++) {
            data[i] = dSigmoid(data[i]);
        }
    }

    static Matrix ApplySigmoid(Matrix& m) {
        Matrix output(m.rows, m.cols);
        for (int i = 0; i < m.rows * m.cols; i++) {
            output.data[i] = Sigmoid(m.data[i]);
        }
        return output;
    }

    static Matrix ApplySigmoidDerivative(Matrix& m) {
        Matrix output(m.rows, m.cols);
        for (int i = 0; i < m.rows * m.cols; i++) {
            output.data[i] = dSigmoid(m.data[i]);
        }
        return output;
    }

    // Functions to convert from 1d array to matrix and from Matrix to 1d array
    // Returns (1 col. Matrix) column vector of 'arr.length' no of rows and 1 column
    static Matrix fromArr(vector<float>& arr) {
        Matrix output(arr.size(), 1);
        for (int i = 0; i < output.rows; i++) {
            output.data[i] = arr[i];
        }
        return output;
    }

    // Load Matrix elements into array of size [rows*cols]
    vector<float> toArr() {
        return data;
    }
};


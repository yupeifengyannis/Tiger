#include <iostream>
#include "Eigen/Dense"

int main(){
    Eigen::MatrixXd mat_a = Eigen::MatrixXd::Ones(4, 4);
    std::cout << mat_a << std::endl; 
}

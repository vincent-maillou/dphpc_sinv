/*
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-09

Copyright 2023 under ETH Zurich DPHPC project course. All rights reserved.
*/

#include <stdio.h>
#include <stdlib.h>
#include <complex>

#include <Eigen/Dense>

#include "utils.h"
#include "system_solve_benchmark.h"

int main() {
    if(!benchmark()){
        printf("Error: benchmark_manasa failed\n");
    }

    return 0;
}









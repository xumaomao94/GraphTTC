# Graph Regularized Tensor Train Completion (GraphTTC)

## Introduction

This is a matlab implemenation of the graph regularized tensor train completion, which optimizes one TT core fiber in each subproblem instead of a TT core. Both GraphTT-opt and GraphTT-vi are included, along with a demo on image completion.

## Functions

- demo_image_completion.m

    Run this demo to test GraphTT-opt/vi on image completion. The adopted image is "TestImages/airplane.mat".

- f_graphTT_opt/

    Includes functions implementing GraphTT-opt.

    - ttc_graph
    
        Use this function to run GraphTT-opt.

- f_graphTT_vi/

    Includes functions implementing GraphTT-VI.
    
    - VITTC_gh
    
        Use this function to run GraphTT-VI.

- rely/

    Includes an implementation on khatrirao product from MATLAB Tensor Toolbox.

- f_perfevaluate/

    Includes functions that evaluate the performance of the recovered tensor.

- TestImages/

    A 'jellybeans' image, and a mask with 80% entries missing

- ExperimentResults/

    A folder used for storing results.

### Reference

Xu, L., Cheng, L., Wong, N., & Wu, Y. C. (2022) To Fold or not to Fold: Graph Regularized Tensor Train for Visual Data Completion. Paper link shall be provided later.

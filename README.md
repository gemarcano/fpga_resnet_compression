# Compressing neural networks for use in FPGAs

## Introduction

Artificial neural networks (ANNs) are being deployed to solve difficult real-world problems, including object identification in images and video, voice recognition, and autonomous driving. The problem with typical ANNs is that they require a lot of computational power to run, limiting their deployment to systems that can host power-hungry graphic processing units (GPUs). Field programmable gate arrays (FPGAs) could be used as an alternative to GPUs for running ANNs with lower power requirements, but FPGAs tend to be resource constrained, making it difficult to port ANNs to them. One solution to this problem is improving neural network compression techniques to help downsize larger neural networks to fit into FPGAs.

This project explores the use of a specific algorithm to help compress residual neural networks (ResNets). Specifically, ResNets have skip connections between their layers which improve the network's capacity to converge on a solution, but adds complexity to FPGA implementations. Prior work done at UC San Diego began to explore the effects of removing skip connections and quantization of network weights on classification accuracy. We build off that work and evaluate the impact on classification accuracy of a special application of knowledge distillation to train ResNets with skip connections removed iteratively, and of quantizing the network with different parameters.

Our implementation leverages Keras and QKeras to implement the neural networks and related training routines, and hls4ml and Vivado HLS to convert ResNets and their skip-less variants to FPGA bitstreams. Our initial results indicate that there is little to no difference between the iterative approach for removing skips versus training the skip-less network normally. We hypothesize that we may be encountering a limitation in Keras, specifically of Tensorflow, that prevented us from implementing the iterative approach in a single training pass, as was the case for prior work.

## Team

 - Doheon Lee - Senior undergraduate student
 - Gabriel Marcano - PhD student

This project was performed for CSE 145/237D under the direction of Ryan Kastner, PhD., and in collaboration with Olivia Weng and Alireza Khodamoradi.

## Layout

This repository contains artifacts related to the project and presentations. Contact Gabriel Marcano via gmarcano@ucsd.edu to gain access to the code repository.

The `Documents` folder contains all of the reports made for the project. The project specification outlines the initial idea of the project, which evolved as we gained a better understanding over machine learning and as we implemented the iterative skip removal algorithm for compressing ResNets. The second document, the milestone document, outlines the updated goals for our project. The final report will be placed here when completed, and will contain a summary of all of our work this quarter, our results, and discussions about why our data disagrees with prior work, as well as possible remedies and work to continue.

The `Presentations` folder contains all of the presentations made for the project, along with our speaker notes. The project overview presentation was prepared at the beginning of the quarter and was meant to be a pitch motivating our project. The mid quarter presentation shows the initial work and results as of the middle of the academic quarter. The final presentation and video will be posted here once they are completed.

## Additional details

![ResNet skip example](./Pictures/ResNet_skip.png)

## For more information

We used the following to learn about 

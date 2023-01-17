# Causal detector : detecting causal agent in a simulated piedestrian trajectory

Machine learning models are increasingly prevalent in trajectory prediction and motion planning tasks for autonomous vehicles (AVs), therefore it is of a big importance to ensure that these models have robust and reliable predictions across a variety of scenarios. However, obtaining the data needed to evaluate and improve the robustness of these models can be difficult and expensive. 

Waymo authors in the paper _CausalAgents: A Robustness Benchmark for Motion Forecasting_ propose a solution to this problem by perturbing existing data by removing certain agents from the scene, which allows them to evaluate and improve the models' robustness to spurious features. In this work, we will focus on the labelling process of these solution, and try to build a classifier that can label an agent as causal or non-causal the ego agent in piedestrian trajectory scene.

## Generate the Data file

This work has been done using generated data from RVO2 based simulator. The labels and framework of the data come from the following github repo,  taht contains the codebase to create and visualizes the synthetic dataset with causality labels (Synth v1), generated using a modified version of the RVO2 simulator.
https://github.com/YuejiangLIU/syntheticmotion/tree/add-causality-labels

Our work has been done on the synth_v1.a.filtered.pkl file from the precedent repo. You can download it following the instructions or generate your own scenes, and put it in the same directory as the notebook. The repo is very well documented and self explenatory.

## Recreate the results

To recreate the results of the report, you only have to follow the documented notebook.

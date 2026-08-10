# EXPLANATION SRC FOLDER STRUCTURE

# Environment Subsystem (env folder)
This folder contains the strict logic and rules of the world the agent interacts with.

# Oracle Data Generation (oracle folder)
This folder is strictly for creating the synthetic dataset and will not be used during neural network training.

# Neural Network Architecture (newtork folder)
This folder isolates the PyTorch model definition.

# Training Engines (training folder)
This folder contains the execution scripts that update the network's weights.

# Shared Utilities (utils folder)
This folder holds helper functions accessed by multiple different phases of the project to ensure data consistency.

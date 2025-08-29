# NO4FSI: Neural Operators for Fluid-Structure Interaction

This repository contains a benchmark implementation of neural operators for learning fluid-structure interaction (FSI) dynamics. The project demonstrates the application of deep learning techniques to predict complex fluid dynamics around moving structures.

## Overview

Fluid-structure interaction problems are fundamental in many engineering applications, from aircraft design to biomedical devices. Traditional computational fluid dynamics (CFD) methods can be computationally expensive and time-consuming. This project explores the use of neural operators as an alternative approach to learn and predict FSI dynamics efficiently.

## Features

- **Multiple Neural Operator Architectures**: Implements three different approaches:
  - **Concat Model**: Concatenates input features for multi-step prediction
  - **Iterative Model**: Uses iterative refinement for improved accuracy
  - **Pooling Model**: Employs pooling strategies for feature aggregation

- **Multi-physics Prediction**: Predicts velocity fields (u, v) and pressure (p) simultaneously

## Architecture

The project implements neural operators that learn mappings between:
- **Input**: Initial fluid state and boundary conditions
- **Output**: Time-evolving fluid velocity and pressure fields

The models use coordinate encoding and boundary condition mapping to handle the complex geometry and physics of FSI problems.

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python main.py --phase train --model concat --data random --num_epochs 20
```

### Testing

```bash
python main.py --phase test --model concat --data random
```

## Model Options

- `--model`: Choose between `concat`, `iterative`, or `pooling`
- `--backend`: Neural network backend (default: `Unet`)
- `--pred_len`: Prediction length in time steps
- `--train_method`: Training method (`normal` or `adaptive`)

## Data

The project works with HDF5/JLD2 format data containing:
- Fluid velocity and pressure time histories
- Boundary condition information
- Structural motion data





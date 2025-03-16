# DeCon Nets: Destruction-Construction Networks

A specialized neural network architecture for repairing noisy 2D tile grids using a novel curriculum learning approach.

## Overview

DeCon Nets implements a dual-network architecture consisting of:
1. A Construction Network that learns to repair noisy 2D tile grids
2. A Destruction Network that predicts the Construction Network's entropy

The system uses curriculum learning to incrementally increase the difficulty of repair tasks based on the Construction Network's performance.

## Features

- Curriculum learning with adaptive difficulty progression
- Dual-network architecture for repair and uncertainty prediction
- Integration with PCGRL (Procedural Content Generation via Reinforcement Learning) environment
- Comprehensive logging and visualization using Weights & Biases
- Checkpoint saving and model persistence

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/decon-nets.git
cd decon-nets
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

To train the networks:

```bash
python -m src.main
```

The training process will:
1. Start with a low difficulty level (5% noise)
2. Train the Construction Network to repair noisy grids
3. Train the Destruction Network to predict entropy
4. Progressively increase difficulty based on performance
5. Save checkpoints and final models

## Project Structure

```
decon-nets/
├── src/
│   ├── __init__.py
│   ├── models.py          # Neural network architectures
│   ├── pcgrl_wrappers.py  # Environment wrappers
│   ├── train.py          # Training loop implementation
│   ├── config.py         # Configuration parameters
│   └── main.py           # Entry point
├── checkpoints/          # Saved model checkpoints
├── models/              # Final trained models
├── logs/               # Training logs
├── requirements.txt    # Project dependencies
└── README.md          # This file
```

## Configuration

Key parameters can be modified in `src/config.py`:
- Initial and maximum difficulty levels
- Network architectures
- Training hyperparameters
- Logging settings

## Results

The system aims to achieve:
- Construction Network repair accuracy: >95% at final difficulty level
- Destruction Network entropy prediction accuracy: MSE < 0.1
- Smooth curriculum progression
- Effective generalization to unseen noise patterns

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
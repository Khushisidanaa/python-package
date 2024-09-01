# MLuno: Univariate Regression Python Package

## Overview

MLuno is a Python package designed for performing univariate regression tasks in machine learning. The package provides tools to simulate data, train models, make predictions, calculate metrics, and visualize results. The focus is on simplicity and clarity, allowing users to perform regression analysis from scratch using minimal dependencies.

## Features

- **Data Simulation**: Generate synthetic data suitable for regression tasks.
- **Data Splitting**: Easily split data into training and testing sets.
- **Regression Models**: Implement basic univariate regression models.
- **Prediction**: Make predictions with trained models.
- **Metrics Calculation**: Evaluate model performance using common metrics.
- **Visualization**: Plot regression results and visualize model performance.

## Getting Started

### Prerequisites

Ensure you have Python installed. You can install the required dependencies with:

```bash
pip install -r requirements.txt
```

### Installation

To install the package, navigate to the root directory of the project and run:

```bash
pip install .
```

### Usage

You can start using MLuno by importing the package and utilizing its modules:

```python
import mluno

# Example usage:
data = mluno.data.make_line_data()
model = mluno.regressors.LinearRegressor()
model.fit(data['X_train'], data['y_train'])
predictions = model.predict(data['X_test'])
mluno.plot.plot_predictions(data['X_test'], predictions)
```

## Directory Structure

Your project directory should look like this:

```kotlin
./mluno/
│
├── src/
│   │
│   └── mluno/
│       ├── __init__.py
│       ├── conformal.py
│       ├── data.py
│       ├── metrics.py
│       ├── plot.py
│       └── regressors.py
│
├── tests/
│   ├── test_conformal.py
│   ├── test_data.py
│   ├── test_metrics.py
│   ├── test_plot.py
│   └── test_regressors.py
│
├── _quarto.yml
├── .gitignore
├── .python-version
├── index.qmd
├── pyproject.toml
└── README.md
```

- **src/mluno/**: Contains the source code for the package.
- **tests/**: Contains unit tests for each module.
- **\_quarto.yml, index.qmd**: Files for generating documentation with Quarto.
- **pyproject.toml**: Defines package dependencies and build system.
- **.gitignore**: Specifies files and directories to be ignored by Git.

## Modules

- **data.py**: Functions for data generation and splitting.
- **regressors.py**: Implementations of various regression models.
- **metrics.py**: Functions for calculating performance metrics.
- **plot.py**: Visualization tools for regression results.
- **conformal.py**: (Optional) Implements split conformal prediction.

## Documentation

MLuno includes comprehensive documentation generated using `quartodoc` and `quarto`. Every function and class is documented with `numpydoc` style docstrings.

To build the documentation, run:

```bash
rye run quartodoc build
quarto preview
```

## Testing

Unit tests are provided to ensure the correctness of the package. Tests can be run using `pytest`:

```bash
rye test
```

## License

This project is open-source and available under the MIT License.


# Web Traffic Forecasting

## Overview
This project aims to predict web traffic or the number of sessions in the next hour based on historical data. Understanding web traffic patterns is crucial for dynamically managing resources and ensuring that websites can efficiently handle the number of visitors at different times.

## Getting Started
Clone this repository to your local machine to get started with the Web Traffic Forecasting project.

### Prerequisites
Ensure you have Python installed on your system. This project uses Python 3.

### Installation
Set up a virtual environment and install the dependencies with the following commands:

```sh
python3 -m venv myenv
source myenv/bin/activate  # On macOS/Linux
myenv\Scripts\activate  # On Windows


### Dependencies
The project depends on several Python libraries listed in the `requirements.txt` file to ensure reproducibility and consistent environments across different setups. These libraries are essential for data processing, modeling, and visualization tasks within the project.

Here is a list of the main libraries and their purpose:

- `pandas`: Used for data manipulation and analysis.
- `numpy`: Adds support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays.
- `matplotlib`: Provides visualization tools to plot data, which is helpful for analyzing web traffic trends.
- `scikit-learn`: Offers various tools for machine learning and statistical modeling including classification, regression, clustering, and dimensionality reduction.
- `tensorflow`: An end-to-end open-source platform for machine learning that enables building and training of neural network models.

To install these dependencies, activate your virtual environment and run the following command:

```sh
pip install -r requirements.txt

```

### Usage
The main scripts for the project are located in the `scripts` directory. To run the forecasting script, execute:

```sh
python scripts/time_series_web_site_forecast.py
```

### Data
The dataset used for modeling is a CSV file containing historical web traffic data, structured with session counts. The path to the dataset is specified in the scripts.

## Project Structure
- `notebook/`: Contains Jupyter notebooks with exploratory data analysis and model prototyping.
- `scripts/`: Contains Python scripts for training and evaluating the forecasting models.
- `myenv/`: Virtual environment directory for project dependencies.
- `best_model.hdf5`: Saved model weights after training.
- `requirements.txt`: List of Python package dependencies for the project.

## Model Architecture
The forecasting model is built using a Sequential LSTM network, suitable for time series data. The model consists of LSTM and Dense layers and is compiled with MSE loss and the Adam optimizer.

## Forecasting
The model predicts the next 24 hours of traffic based on the previous week's data. A custom forecasting function processes the input data and generates future traffic predictions.

## Visualization
The `plot` function provides a visual comparison between the true traffic values and the predictions, aiding in performance evaluation.

## License
This project is licensed under the MIT License.

## Acknowledgments
- Special thanks to all contributors and maintainers of the packages used in this project.



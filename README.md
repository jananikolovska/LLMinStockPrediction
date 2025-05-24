# AI for stock market prediction: Using LLMs for TimeSeries Predictions

Project by: Jana Nikolovska <br>
Supervised by: Giacomo Frisoni, MSc <br>
Prof. Gianluca Moro, PhD <br>

ALMA MATER STUDITORIUM - University of Bologna
May 2025

---
**Summary:** <br>
In this project, I explore the use of Large Language Models (LLMs) for time series forecasting, focusing on the task of stock market prediction. The work was proposed and mentored by Prof. Gianluca Moro and Giacomo Frisoni at the University of Bologna.

As a starting point, I used a provided notebook as starting point . The notebook introduces the dataset (historical S&P 500 data via [`yfinance`](https://pypi.org/project/yfinance/)), a baseline linear regression model for comparison, and the *Trading Protocol* — a framework to evaluate forecasting performance by simulating trading strategies.

For the LLM-based forecasting approach, I followed the methodology described in [this paper](https://arxiv.org/pdf/2310.07820) and its [official implementation](https://github.com/ngruver/llmtime/tree/main). The code from the paper has been adapted and extended with additional functionality tailored to the specific requirements of my experiments.

_Goal_: <br>
The goal of this project was to create a similar problem to those of the referenced paper, particularly in terms of train and test sizes. The problem was defined such that with 150 days of value history, the goal was to predict the subsequent 29 days (for both open and closed time series), following an autoregressive approach without using ground truth. 
* Modifications were made to the Linear Regression model from the baseline notebook. While it still utilizes a lagged dataset for predictions, a simulation of autoregressiveness was incorporated to make it more comparable to autoregressive models. <br>
The dataset was split into multiple 150-29 train-test sections, and models were trained and evaluated independently on each split. <br>

_Evaluation_: <br>
For evaluation, RMSE and MAPE were used to assess the models' predictive accuracy, while a trading protocol was employed to simulate trading and profit. To enhance the results, I averaged the outcomes across the different splits to achieve a more reliable measure of model performance. Various visualizations were included throughout the notebook to enhance the understanding of the results."

### Overview
`DATA_ANALYSIS.ipynb` Plotly visualizations and analysis of time series data <br/>
`PROJECT.ipynb` Inference and evaluation pipeline <br/>
`PRESENTATION.pdf` short PowerPoint Presentation <br/>

### Replicability 

#### Step 1: Installation
**(Linux)** Run the following command to install all dependencies in a conda environment named llm_stock_prediciton. Change the cuda version for torch if you don't have cuda 11.8.

```bash
source install.sh
```

After installation, activate the environment with
```bash
conda activate llm_stock_prediciton
```
**(Windows)** Run the following command to install all dependencies in a conda environment named llm_stock_prediciton. Change the cuda version for torch if you don't have cuda 11.8.
```bash
conda create -n llm_stock_prediciton python=3.9
conda activate llm_stock_prediciton
pip install -r requirements.txt
```
If you prefer not using conda, you can also install the dependencies listed in install.sh manually.

#### Step 2: Create a `secrets` Folder

In the root of your project directory, create a folder named `secrets`:

```bash
mkdir secrets
```

#### Step 3: Add the API Key

Inside the `secrets` folder, create a file called `openai_key.txt`:

```bash
touch secrets/openai_key.txt
```

Paste your OpenAI API key into that file. It should look like this:

```
sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```
#### Step 4: Run Jupyter Notebook
```
jupyter notebook
```
Open and execute either `DATA_ANALYSIS.ipynb` or `PROJECT.ipynb`

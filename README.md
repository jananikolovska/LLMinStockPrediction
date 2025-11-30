
# AI for stock market prediction: Using LLMs for TimeSeries Predictions (LLMTime implementation)

Project by: Jana Nikolovska <br>
Supervised by: Giacomo Frisoni, MSc <br><!--  -->
Prof. Gianluca Moro, PhD <br>

ALMA MATER STUDITORIUM - University of Bologna <br>
November 2025

---

**Summary:** <br>
In this project, I explore the use of Large Language Models (LLMs) for time series forecasting, focusing on the task of stock market prediction. The work was proposed and mentored by Prof. Gianluca Moro and Giacomo Frisoni at the University of Bologna.

As a starting point, I used a provided notebook by my mentors. The notebook introduces the dataset (**historical S&P 500 data via [`yfinance`](https://pypi.org/project/yfinance/)**), a basic linear regression model, which I chose not to include in the final results, that served as an early benchmark and highlighted the underlying complexity of the forecasting problem and finally, the **"Trading Protocol"** — a framework designed to evaluate forecasting performance through simulated trading strategies —that I extended upon to incorporate multiple strategy types and Monte Carlo-based robustness testing.

On the chosen dataset, a traditional **statistical model ARIMA** and **LLM-based forecasting approaches** have been tested and evaluated. For the implementation for **LLM-Time**, I followed the methodology described in [this paper](https://arxiv.org/pdf/2310.07820) and its [official implementation](https://github.com/ngruver/llmtime/tree/main). The code from the paper has been adapted and extended with additional functionality tailored to the specific requirements of my experiments. I also included a comparison with **TimesFM** following the example code from [this blog]() and [code repository]().

The problem was defined comparable to that of the referenced paper, particularly in terms of training and testing proportions. The task was defined such that, given 150 days of historical values, the objective was to predict the subsequent 29 days values following an autoregressive approach without direct access to ground truth during prediction.

Unlike the provided baseline notebook, which operated directly on price levels, this project instead utilized log returns as the primary time-series input. This choice was motivated by findings from the exploratory data analysis, which indicated that the raw price series exhibited non-stationary behavior — making a dificult target for predicting. Transforming the data into returns helped achieve approximate stationarity, giving a chance to  the models to capture relative changes more effectively.

As mentioned earlier, both statistical and trading-based evaluation methods were used, with greater emphasis placed on the trading metrics (Trading Protocol), while the statistical ones were included primarily for completeness and exploratory insight.


### Overview
`DATA_ANALYSIS.ipynb` Plotly visualizations and analysis of time series data <br/>
`PROJECT.ipynb` Inference and evaluation pipeline for **ARIMA** and **LLMTime**. Combined results including TimesFM <br/>
`PROJECT_timesfm.ipynb` Inference and evaluation pipeline for **TimesFM** (needed to be separated due to different preconditions and environments)
`PRESENTATION.pdf` short PowerPoint Presentation <br/>

### Replicability 

The steps for replicability refer to the `PROJECT.ipynb` and `DATA_ANALYSIS.ipynb` notebook, whereas the `PROJECT_timesfm.ipynb` can be run as it is on Google Colab

#### Step 1: Installation
_Disclaimer: it needs to be run on Linux or WSL_ <br/>
Run the following command to install all dependencies in a conda environment named llm_stock_prediciton. Change the cuda version for torch if you don't have cuda 11.8.

```bash
source install.sh
```

After installation, activate the environment with
```bash
conda activate llm_stock_prediciton
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

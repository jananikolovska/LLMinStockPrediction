# LLMinStockPrediction

**Summary:** <br>
In this project, I explore the use of Large Language Models (LLMs) for time series forecasting, focusing on the task of stock market prediction. The work was proposed and mentored by Prof. Gianluca Moro and Giacomo Frisoni at the University of Bologna.

As a starting point, I used a provided notebook that introduces the dataset (historical S&P 500 data via [`yfinance`](https://pypi.org/project/yfinance/)), a baseline linear regression model for comparison, and the *Trading Protocol* — a framework to evaluate forecasting performance by simulating trading strategies.

For the LLM-based forecasting approach, I followed the methodology described in [this paper](https://arxiv.org/pdf/2310.07820) and its [official implementation](https://github.com/ngruver/llmtime/tree/main). The code from the paper has been adapted and extended with additional functionality tailored to the specific requirements of my experiments.

In addition to implementing these models, I conducted further analysis of the stock time series itself as well as *[to be completed]*.

### Replicability 

#### Step 1: Installation
Run the following command to install all dependencies in a conda environment named llmtime. Change the cuda version for torch if you don't have cuda 11.8.

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
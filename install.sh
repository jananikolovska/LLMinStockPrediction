conda create -n llm_stock_prediciton python=3.9
conda activate llm_stock_prediciton
pip install numpy
pip install -U jax[cpu] # we don't need GPU for jax
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install openai==0.28.1
pip install tiktoken
pip install tqdm
pip install matplotlib
pip install "pandas<2.0.0"
pip install darts
pip install gpytorch
pip install transformers
pip install datasets
pip install multiprocess
pip install SentencePiece
pip install accelerate
pip install gdown
pip install dash
pip install plotly
pip install jupyter
pip install mistralai
pip install seaborn
pip install -U kaleido
conda deactivate

import matplotlib as plt
import pandas as pd
import numpy as np
from models.utils import grid_iter
from models.validation_likelihood_tuning import get_autotuned_predictions_data

def plot_preds(train, test, pred_dict, model_name, show_samples=False, hash_function=None):
    """
        Function adapted from: https://github.com/ngruver/llmtime
        Original author(s): Nicholas Gruver et al.
    """
    pred = pred_dict['median']
    pred = pd.Series(pred, index=test.index)
    plt.figure(figsize=(8, 6), dpi=100)
    plt.plot(train)
    plt.plot(test, label='Truth', color='black')
    plt.plot(pred, label=model_name, color='purple')
    # shade 90% confidence interval
    samples = pred_dict['samples']
    lower = np.quantile(samples, 0.05, axis=0)
    upper = np.quantile(samples, 0.95, axis=0)
    plt.fill_between(pred.index, lower, upper, alpha=0.3, color='purple')
    if show_samples:
        samples = pred_dict['samples']
        # convert df to numpy array
        samples = samples.values if isinstance(samples, pd.DataFrame) else samples
        for i in range(min(10, samples.shape[0])):
            plt.plot(pred.index, samples[i], color='purple', alpha=0.3, linewidth=1)
    plt.legend(loc='upper left')
    if 'NLL/D' in pred_dict:
        nll = pred_dict['NLL/D']
        if nll is not None:
            plt.text(0.03, 0.85, f'NLL/D: {nll:.2f}', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    if hash_number:
        filename = os.path.join(RESULT_IMAGES_PATH, f'plot_{hash_number}.png')
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

def get_inference(data, ds_name, num_samples=10, model_names=None, model_hypers=None, model_predict_fns = None, visualize=True):
    """
        Function adapted from: https://github.com/ngruver/llmtime
        Original author(s): Nicholas Gruver et al.
    """
    train, test = data # or change to your own data
    out = {}
    for model in model_names: # GPT-4 takes a about a minute to run
        model_hypers[model].update({'dataset_name': ds_name}) # for promptcast
        hypers = list(grid_iter(model_hypers[model]))
        pred_dict = get_autotuned_predictions_data(train, test, hypers, num_samples, model_predict_fns[model], verbose=False, parallel=False)
        out[model] = pred_dict
        if visualize:
            plot_preds(train, test, pred_dict, model, show_samples=True)
    return model, out

def process_prediction_samples(samples):
    # Summarizes multiple predictions by sampling from their mean and std to capture uncertainty
    mu, std = np.mean(samples), np.std(samples)
    sampled_value = np.random.normal(mu, std)
    return sampled_value

def process_prediction_samples(samples):
    ss = []
    for col in samples.columns:
        mu = samples[col].mean()
        sigma = samples[col].std(ddof=1)

        sample = np.random.normal(mu, sigma)  # use this if you want to draw a random sample instead
        ss.append(mu) #sample or mu

    return ss
    
def process_llmtime_outputs(open_samples, close_samples):
    sampled_open_predictions = process_prediction_samples(open_samples)
    sampled_close_predictions = process_prediction_samples(close_samples)
    return sampled_open_predictions, sampled_close_predictions
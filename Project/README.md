# Stocks Closing Movements Prediction

The project is one of Kaggle's competition 
[Optiver - Trading at the Close](https://www.kaggle.com/competitions/optiver-trading-at-the-close/overview).

## Data

you can download it via our [Google Drive](https://drive.google.com/file/d/1zDIahTfTqsnWU3Yn-SJahh111YcGpkrL/view?usp=sharing).

Or join the competition and download the data on [Kaggle](https://www.kaggle.com/competitions/optiver-trading-at-the-close/data).




## Code

We only provide training code here since inference code needs `optiver2023` package that is available 
on Kaggle platform only.

You need to change the input/output paths of data to your local paths, for example, in `mlp_baseline.ipynb`,
you need to change the following input path.
```python
train = pd.read_csv('/kaggle/input/optiver-trading-at-the-close/train.csv')
```

You can access our code through local files in the repository or on the Kaggle platform (if we had set the code public). 

### Data Analysis & Processing

- Data Analysis & Processing
  - local: `data_analysis.ipynb`
  - Kaggle: [here](https://www.kaggle.com/code/ruihaokaggle/data-analysis)


### Model

- LightGBM
  - local: `lightgbm_baseline.ipynb`

- XGBoost
  - local: `xgboost_baseline.ipynb`
  - Kaggle: private
- CatBoost
  - local: `catboost_baseline.ipynb`
  - Kaggle: [here](https://www.kaggle.com/code/ruihaokaggle/catboost-kf-baseline)
- MLP + AutoEncoder
  - local: `mlp_baseline.ipynb`
  - Kaggle: [here](https://www.kaggle.com/code/ruihaokaggle/mlp-baseline?scriptVersionId=148528670)
- CNN: 
  - local: `1dcnn_baseline.ipynb`
  - Kaggle: [here](https://www.kaggle.com/code/ruihaokaggle/1dcnn-baseline?scriptVersionId=148500474)
- Transformer: 
  - local: `transformer_baseline.ipynb`
  - Kaggle: [here](https://www.kaggle.com/code/ruihaokaggle/simple-transformer-optiver-trading-at-the-close?scriptVersionId=151312998)
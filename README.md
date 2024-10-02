# Sepsis-AMR-JAMA

Code for the paper "Performance of risk models for antimicrobial resistance in adult patients with sepsis".

### Setup
```
pip install torch torchvision networkx node2vec xgboost shap scikit-learn
```
Change paths in file ```.paths.py```

### Data preprocessing
```
cd preprocess_cohort_3
python generate_labels.py
python preprocess_raw.py
python prepare_data_simple.py
python prepare_data_deep.py
```

### Run deep model
```
python run_deep.py -c configs/deep_model.yaml
```

### Run XGBoost model
```
python run_simple_ml.py
```

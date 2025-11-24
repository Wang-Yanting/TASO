# TASO: Jailbreak LLMs via Alternative Template and Suffix Optimization

This repository provides the official PyTorch implementation of TASO: Jailbreak LLMs via Alternative Template and Suffix Optimization. TASO (Template and Suffix Optimization) is a novel jailbreak method that optimizes both a template and a suffix in an alternating manner.



### 🔨 Setup environment

Please run the following commands to set up the environment:

```bash
conda env create TASO
conda activate TASO
pip install -r requirements.txt
```
### 🔑 Set API key

Please set up your api key as an enviornment variable:

```bash
export HF_TOKEN="your_hf_token_here"
export OPENAI_API_KEY="your_openai_token_here"
```

### 🔬 Experiments

Here we provide the scripts to replicate our experimental findings.
To run our method, you can use the python script [script.py](script.py).

You can set the argumetn `target_model` to the model you want to use. The model names can be found in [models.yaml](configs/model_configs/models.yaml). If it is a closed-source model, you need to change the method to `TASO_blackbox`(the default method is `TASO_whitebox`).
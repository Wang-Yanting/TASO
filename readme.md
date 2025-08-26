# TracLLM: A Generic Framework for Attributing Long Context LLMs

<p align='center'>
    <img alt="TASO" src='assets/fig1.png' width='50%'/>

</p>

This repository provides the official PyTorch implementation of TASO: Jailbreak LLMs via Alternative Template and Suffix Optimization.

TASO (Template and Suffix Optimization) is a novel jailbreak method that optimizes both a template and a suffix in an alternating manner.



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
```

### 🔬 Experiments

Here we provide the scripts to replicate our experimental findings.
To run our method, you can use the following script:
- [script.py](script.py)

You can set "target_model" to the model you want to use. The model names can be found in - [models.yaml](configs/model_configs/models.yaml). If it is a closed-source model, you need to change the method to "TASO_blackbox"(the default method is "TASO_whitebox").
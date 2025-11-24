from utils.utils import load_harmful_dataset
import os
import pandas as pd

method = "TASO_whitebox"
target_model = "llama2_7b"  #"qwen_7b_chat" 
dataset_name = "harmbench"
gpu_id=0
n_responses = 5
prompt_template = "default"
n_constraints = 5
n_constraint_updates = 10
if "gpt" in target_model:
    n_iterations =100
else:
    n_iterations = 1000

attacker_model = "gpt-4"
n_tokens_adv = 50
log_dir = f'logs/{target_model}/{dataset_name}/{method}'
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, f'{dataset_name}_{method}_{attacker_model}_{n_responses}_{n_constraints}_{n_constraint_updates}_{n_iterations}_{prompt_template}.log.out')

cmd = f'python3 -u main.py \
--method {method} \
--dataset_name {dataset_name} \
--n-iterations {n_iterations} \
--prompt-template "{prompt_template}" \
--target-model "{target_model}" \
--attacker-model "{attacker_model}" \
--judge-model gpt-4o-mini \
--n-tokens-adv {n_tokens_adv} \
--n-tokens-change-max 4 \
--schedule_prob \
--judge-max-n-calls 5 \
--n-restarts 5 \
--n-responses {n_responses} \
--n-constraints {n_constraints} \
--n-constraint-updates {n_constraint_updates} \
--target-max-n-tokens 500 \
--determinstic-jailbreak \
--gpu_id {gpu_id} \
2>&1 | tee -a {log_path} '
result = os.system(cmd)
print(result)

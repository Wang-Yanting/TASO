
import torch
import os
from typing import List
from language_models import GPT, GPT_azure, HuggingFace
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from configs.model_configs.config import LLAMA_7B_PATH, LLAMA_13B_PATH,LLAMA3_13B_PATH,LLAMA31_13B_PATH, LLAMA_70B_PATH, LLAMA3_8B_PATH,LLAMA31_8B_PATH, LLAMA3_70B_PATH,LLAMA31_70B_PATH, QWEN_7B_PATH, GEMMA_2_9B_PATH, MISTRAL_7B_PATH, TARGET_TEMP, TARGET_TOP_P
import yaml
import time
from fastchat.model import get_conversation_template
from utils.string_utils import get_template

def load_target_model(args):
    targetLM = TargetLM(args,model_name = args.target_model, 
                        temperature = TARGET_TEMP, # init to 0
                        top_p = TARGET_TOP_P, # init to 1
                        use_logit_lens =False,
                        device = args.gpu_id
                        )
    return targetLM
class TargetLM():
    """
    Base class for target language models.
    
    Generates responses for prompts using a language model. The self.model attribute contains the underlying generation model.
    """
    def __init__(self, args,
            model_name: str, 
            temperature: float,
            top_p: float,
            use_logit_lens: bool = False,
            device: int = 0
            ):
        self.args = args
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.model,template_name= load_indiv_model(model_name, device,use_logit_lens = use_logit_lens)
        config_dir = args.chat_config
        with open(config_dir) as file:
            model_configs = yaml.full_load(file)
        target_model = model_configs[model_name]['model']
        if "gpt" not in self.model_name:
            template = get_template(**target_model)['prompt']
            self.msg_start, self.msg_end = template.split("{instruction}")

            self.template = template
        self.template_name = template_name
        self.n_input_tokens = 0
        self.n_output_tokens = 0
        self.n_input_chars = 0
        self.n_output_chars = 0

    def get_response(self, prompts_list: List[str], max_n_tokens=None, temperature=None) -> List[dict]:

        full_prompts = []
        
        for prompt in prompts_list:
            if "gpt" not in self.model_name:
                full_prompts.append(self.msg_start+prompt+self.msg_end)
            else:
                conv=get_conversation_template(self.template_name)
                conv.append_message(conv.roles[0], prompt)
                full_prompts.append(conv.to_openai_api_messages())
        with torch.no_grad():
            print("generating...")
            outputs = self.model.generate(full_prompts, 
                                        max_n_tokens=max_n_tokens,  
                                        temperature=self.temperature if temperature is None else temperature,
                                        top_p=self.top_p
            )
        return outputs


    def initialize_kv_cache(self, prompts_list: List[str], no_template=False) -> List[dict]:
        msg_start, msg_end = self.msg_start, self.msg_end
        full_prompts = []  # batch of strings
        kv_cache_ids = []
        kv_caches = []  # list to store kv_cache for each prompt
 
        full_prompts = [self.model.tokenizer.encode(msg_start+prompt+msg_end) for prompt in prompts_list]

        kv_cache_ids = full_prompts
        full_prompts=torch.tensor(full_prompts).to(self.model.model.device)
        with torch.no_grad():
            outputs = self.model.model(full_prompts, use_cache=True)

        kv_caches = outputs.past_key_values  # extract kv_cache from the output
        if kv_caches is None:
            return [[]],[[]]
        return tuple(kv_caches),kv_cache_ids  # return the list of kv_caches

    def update_kv_cache(self, kv_cache, kv_cache_ids, new_msg,target_token,no_template=0):

        msg_start, msg_end = self.msg_start, self.msg_end
        msg_token_ids = self.model.tokenizer.encode(msg_start+new_msg+msg_end)

        start_idx = 0
        start_check = len(msg_token_ids)-100
        for i, (a, b) in enumerate(zip(kv_cache_ids[start_check:], msg_token_ids[start_check:]),start = start_check):
            if a != b:
                start_idx = i-10
                break
 
        kv_cache = self.slice_kv_cache(kv_cache, 0,start_idx)
 
        input_ids = torch.tensor([msg_token_ids[start_idx:]]).to(self.model.model.device)


        with torch.no_grad():
            outputs = self.model.model(input_ids=input_ids, past_key_values=kv_cache, use_cache=True)

        kv_cache = outputs.past_key_values
        return kv_cache, [msg_token_ids]
    def get_logprob(self, msg, target_token: str) -> List[dict]:
        msg_start, msg_end = self.msg_start, self.msg_end
        msg_token_ids = self.model.tokenizer.encode(msg_start + msg + msg_end)
        with torch.no_grad():

            input_ids = torch.tensor([msg_token_ids]).to(self.model.model.device)
            outputs = self.model.model(input_ids)

            logits = outputs[0][:, -1, :][0]
            target_token_id = self.model.tokenizer.convert_tokens_to_ids(target_token)
            log_prob = torch.log_softmax(logits, dim=-1)[target_token_id].item()
            #log_probs = torch.log_softmax(logits, dim=-1)[:, target_token_id].tolist()
            top_k = 5
            top_k_probs, top_k_indices = torch.topk(torch.log_softmax(logits, dim=-1), top_k)
            top_k_tokens = self.model.tokenizer.convert_ids_to_tokens(top_k_indices.tolist())
            if "qwen" not in self.model_name.lower():
                log_prob_dict = {token.replace('▁', ' '): prob.item() for token, prob in zip(top_k_tokens, top_k_probs)}
            else:
                log_prob_dict = {token: prob.item() for token, prob in zip(top_k_tokens, top_k_probs)}
            if "orca" in self.model_name.lower():
                print("log_prob_dict: ", log_prob_dict)
                print("start token id: ",self.model.tokenizer.convert_tokens_to_ids('<0x0A>'))
        torch.cuda.empty_cache()
        return log_prob, log_prob_dict
    def get_logprob_kv(self, kv_cache, kv_cache_ids, new_msg, target_token, no_template=0, kv_eviction=False):
        msg_start, msg_end = self.msg_start, self.msg_end
        msg_token_ids = self.model.tokenizer.encode(msg_start + new_msg + msg_end)
        if "orca" in self.model_name:
            msg_token_ids = msg_token_ids  +[13]
        if "gemma2" not in self.model_name.lower():
            start_idx = 0
            start_check = len(msg_token_ids) - 100
            for i, (a, b) in enumerate(zip(kv_cache_ids[start_check:], msg_token_ids[start_check:]), start=start_check):
                if a != b:
                    start_idx = i - 10
                    break

            if kv_eviction:
                kv_cache = self.slice_kv_cache(kv_cache, start_idx - 3, start_idx)
            else:
                kv_cache = self.slice_kv_cache(kv_cache, 0, start_idx)

        if "qwen" in self.model_name.lower() or "gemma2" in self.model_name.lower():
            input_ids = torch.tensor([msg_token_ids]).to(self.model.model.device)
        else:
            input_ids = torch.tensor([msg_token_ids[start_idx:]]).to(self.model.model.device)
        with torch.no_grad():
            start_time = time.time()
            if "qwen" in self.model_name.lower() or "gemma2" in self.model_name.lower():
                outputs = self.model.model(input_ids=input_ids, use_cache=True)
            else:
                outputs = self.model.model(input_ids=input_ids, past_key_values=tuple(kv_cache), use_cache=True)
            end_time = time.time()
            print(f'model time: {end_time - start_time}')
        logits = outputs[0][:, -1, :][0]
        start_time = time.time()
        
        target_token_id = self.model.tokenizer.convert_tokens_to_ids(target_token)
        if "llama2" in self.model_name:
            target_token_id = 18585  # this is the id of "_Sure"
        if "qwen" in self.model_name.lower():
            target_token_id = self.model.tokenizer.convert_tokens_to_ids(b"Sure")
  
        log_prob = torch.log_softmax(logits, dim=-1)[target_token_id].item()

        # Get top-5 tokens and their log probabilities
        top_k = 5
        top_k_probs, top_k_indices = torch.topk(torch.log_softmax(logits, dim=-1), top_k)
        top_k_tokens = self.model.tokenizer.convert_ids_to_tokens(top_k_indices.tolist())
        if "qwen" not in self.model_name.lower():
            log_prob_dict = {token.replace('▁', ' '): prob.item() for token, prob in zip(top_k_tokens, top_k_probs)}
        else:
            log_prob_dict = {token: prob.item() for token, prob in zip(top_k_tokens, top_k_probs)}
        if "orca" in self.model_name.lower():
            print("log_prob_dict: ", log_prob_dict)
            print("start token id: ",self.model.tokenizer.convert_tokens_to_ids('<0x0A>'))
        end_time = time.time()
        print(f'log_softmax time: {end_time - start_time}')

        return log_prob, log_prob_dict

    def slice_kv_cache(self,cache, k1,k2):
        # Create a new cache with the first k token positions
        new_cache = []
        for key, value in cache:
            # Slice the key and value tensors to keep only the first k positions
            new_key = key[:, :, k1:k2, :]
            new_value = value[:, :, k1:k2, :]
            new_cache.append((new_key, new_value))
        return new_cache
    def candidate_control(self, orig_msg,adv_token_ids, target_token, k=100):

        """
        Computes gradients of the loss with respect to the coordinates.
        
        Parameters
        ----------
        model : Transformer Model
            The transformer model to be used.
        input_ids : torch.Tensor
            The input sequence in the form of token ids.
        input_slice : slice
            The slice of the input sequence for which gradients need to be computed.
        target_slice : slice
            The slice of the input sequence to be used as targets.
        loss_slice : slice
            The slice of the logits to be used for computing the loss.

        Returns
        -------
        torch.Tensor
            The gradients of each token in the input_slice with respect to the loss.
        """
        #print('adv_token_ids', len(adv_token_ids))
        msg_token_ids = torch.tensor(self.wrap_system_message_end(self.wrap_system_message_start(self.model.tokenizer.encode(orig_msg)+adv_token_ids))).to(self.model.model.device)
        if "orca" in self.model_name:
            msg_token_ids = torch.tensor(self.wrap_system_message_end(self.wrap_system_message_start(self.model.tokenizer.encode(orig_msg)+adv_token_ids+[13]))).to(self.model.model.device)
 
        # Find the adversarial slice - tokens that appear in new_msg but not orig_msg
        adv_slice = slice(
            len(self.wrap_system_message_start(self.model.tokenizer.encode(orig_msg))), 
            len(self.wrap_system_message_start(self.model.tokenizer.encode(orig_msg)+adv_token_ids))
        )
        #fprint(input_slice.shape)
        print("adv_slice: ",adv_slice.start,adv_slice.stop)

        embed_weights = self.model.model.get_input_embeddings().weight
        one_hot = torch.zeros(
            msg_token_ids[adv_slice].shape[0],
            embed_weights.shape[0],
            device=self.model.model.device,
            dtype=embed_weights.dtype
        )
        one_hot.scatter_(
            1, 
            msg_token_ids[adv_slice].unsqueeze(1),
            torch.ones(one_hot.shape[0], 1, device=self.model.model.device, dtype=embed_weights.dtype)
        )
        one_hot.requires_grad_()
        input_embeds = (one_hot @ embed_weights).unsqueeze(0)
        
        # now stitch it together with the rest of the embeddings
        embeds = self.model.model.get_input_embeddings()(msg_token_ids.unsqueeze(0)).detach()
        full_embeds = torch.cat(
            [
                embeds[:,:adv_slice.start,:], 
                input_embeds, 
                embeds[:,adv_slice.stop:,:]
            ], 
            dim=1)

        logits = self.model.model(inputs_embeds=full_embeds)[0][:, -1, :][0]
        if "llama2" in self.model_name:
            target_token_id = 18585 #this is the id of "_Sure"
        elif "qwen" in self.model_name.lower():
            target_token_id = self.model.tokenizer.convert_tokens_to_ids(b"Sure")
            print("target_token_id: ", target_token_id)
        else:
            target_token_id = self.model.tokenizer.convert_tokens_to_ids(target_token)
        loss = torch.log_softmax(logits, dim=-1)[target_token_id]
        #print('loss', loss.item())
        loss.backward()
        grad = one_hot.grad.clone()
        grad = grad / grad.norm(dim=-1, keepdim=True)
        top_k_tokens = torch.topk(grad, k, dim=-1).indices.tolist()
        #print('top_k_tokens', top_k_tokens)
        return top_k_tokens
    def candidate_control_kv(self, orig_msg_cache,adv_token_ids, target_token, k=10):
        """
        Computes gradients of the loss with respect to the coordinates.
        
        Parameters
        ----------
        model : Transformer Model
            The transformer model to be used.
        input_ids : torch.Tensor
            The input sequence in the form of token ids.
        input_slice : slice
            The slice of the input sequence for which gradients need to be computed.
        target_slice : slice
            The slice of the input sequence to be used as targets.
        loss_slice : slice
            The slice of the logits to be used for computing the loss.

        Returns
        -------
        torch.Tensor
            The gradients of each token in the input_slice with respect to the loss.
        """
        #print('adv_token_ids', len(adv_token_ids))
        if "llama2" not in self.model_name:
            msg_token_ids = torch.tensor(self.wrap_system_message_end(adv_token_ids)).to(self.model.model.device)
        else:
            #msg_token_ids = torch.tensor(self.wrap_system_message_end(adv_token_ids)+[29871]).to(self.model.model.device)
            msg_token_ids = torch.tensor(self.wrap_system_message_end(adv_token_ids)).to(self.model.model.device)
        
        # Find the adversarial slice - tokens that appear in new_msg but not orig_msg
        adv_slice = slice(
            0, 
            len(adv_token_ids)
        )
        #fprint(input_slice.shape)
        print("adv_slice: ",adv_slice.start,adv_slice.stop)

        embed_weights = self.model.model.get_input_embeddings().weight
        one_hot = torch.zeros(
            msg_token_ids[adv_slice].shape[0],
            embed_weights.shape[0],
            device=self.model.model.device,
            dtype=embed_weights.dtype
        )
        one_hot.scatter_(
            1, 
            msg_token_ids[adv_slice].unsqueeze(1),
            torch.ones(one_hot.shape[0], 1, device=self.model.model.device, dtype=embed_weights.dtype)
        )
        one_hot.requires_grad_()
        input_embeds = (one_hot @ embed_weights).unsqueeze(0)
        
        # now stitch it together with the rest of the embeddings
        embeds = self.model.model.get_input_embeddings()(msg_token_ids.unsqueeze(0)).detach()
        full_embeds = torch.cat(
            [
                embeds[:,:adv_slice.start,:], 
                input_embeds, 
                embeds[:,adv_slice.stop:,:]
            ], 
            dim=1)

        logits = self.model.model(inputs_embeds=full_embeds, past_key_values=orig_msg_cache, use_cache=True)[0][:, -1, :][0]
        if "llama2" in self.model_name:
            target_token_id = 18585 #this is the id of "_Sure"
        else:
            target_token_id = self.model.tokenizer.convert_tokens_to_ids(target_token)
        loss = torch.log_softmax(logits, dim=-1)[target_token_id]
        #print('loss', loss.item())
        loss.backward()
        grad = one_hot.grad.clone()
        grad = grad / grad.norm(dim=-1, keepdim=True)
        top_k_tokens = torch.topk(grad, k, dim=-1).indices.tolist()
        #print('top_k_tokens', top_k_tokens)
        return top_k_tokens
    def wrap_system_message_start(self,msg, use_ids = False):
        #return msg
        #msg_start = ""
        msg_start = self.msg_start

        msg_start = self.model.tokenizer.encode(msg_start)
        #print('msg_start', msg_start)
        return msg_start+msg

    def wrap_system_message_end(self,msg,use_ids = False):
        msg_end = self.msg_end

        msg_end = self.model.tokenizer.encode(msg_end)
        msg_end = self.remove_bos_token(msg_end)
        #print('msg_end', msg_end)
        return msg+msg_end
    def remove_bos_token(self, token_ids):
        if token_ids[0] == self.model.tokenizer.bos_token_id:
            token_ids = token_ids[1:]
        return token_ids


def load_indiv_model(model_name, device=None,use_logit_lens = False, azure = False):
    model_path, template_name = get_model_path_and_template(model_name)
    print("device", device)
    if 'gpt' in model_name or 'together' in model_name:
        if azure:
            lm = GPT_azure(model_name)
        else:
            lm = GPT(model_path)

    else:
        print("device", device)
        print("model_path", model_path)
        if '70b' in model_path.lower() or '10_7b' in model_path.lower() or '72b' in model_path.lower() or 'deepseek' in model_path.lower() or '8x7b' in model_path.lower():
            model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                torch_dtype=torch.float16,
                device_map="auto",
                token=os.getenv("HF_TOKEN"),
                trust_remote_code=True).eval()
        else:
            print("device", device)
            if "baichuan" not in model_path.lower():
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, 
                    torch_dtype=torch.float16,
                    device_map="cuda:"+str(device),
                    token=os.getenv("HF_TOKEN"),
                    trust_remote_code=True).eval()
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, 
                    torch_dtype=torch.bfloat16,
                    device_map="cuda:"+str(device),
                    token=os.getenv("HF_TOKEN"),
                    trust_remote_code=True).eval()
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            trust_remote_code=True,
            token=os.getenv("HF_TOKEN")
        )

        if 'llama2' in model_path.lower():
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.padding_side = 'left'
        if 'vicuna' in model_path.lower():
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'
        if 'mistral' in model_path.lower():
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        lm = HuggingFace(model_name, model, tokenizer)
    
    return lm,template_name

def get_model_path_and_template(model_name):
    full_model_dict={
        "gpt-4-0125-preview":{
            "path":"gpt-4-0125-preview",
            "template":"gpt-4"
        },

        "gpt-4-1106-preview":{
            "path":"gpt-4-1106-preview",
            "template":"gpt-4"
        },
        "gpt-4-turbo":{
            "path":"gpt-4-turbo",
            "template":"gpt-4"
        },
        "gpt-4":{
            "path":"gpt-4-1106-preview",
            "template":"gpt-4"
        },
        "gpt-4o":{
            "path":"gpt-4o",
            "template":"gpt-4"
        },
        "gpt-4o-mini":{
            "path":"gpt-4o-mini",
            "template":"gpt-4o-mini"
        },
        "gpt-3.5-turbo": {
            "path":"gpt-3.5-turbo-0125",
            "template":"gpt-3.5-turbo"
        },
        "gpt-3.5-turbo-1106": {
            "path":"gpt-3.5-turbo",
            "template":"gpt-3.5-turbo"
        },
        "gpt-3.5-turbo-0125": {
            "path":"gpt-3.5-turbo-0125",
            "template":"gpt-3.5-turbo-0125"
        },
        "vicuna_7b_v1_5":{
            "path":"lmsys/vicuna-7b-v1.5",
            "template":"vicuna_v1.1"
        },
        "vicuna_13b_v1_5":{
            "path":"lmsys/vicuna-13b-v1.5",
            "template":"vicuna_v1.1"
        },
        "llama2":{
            "path":LLAMA_7B_PATH,
            "template":"llama-2-chat"
        },
        "llama2_7b":{
            "path":LLAMA_7B_PATH,
            "template":"llama-2-chat"
        },
        "llama2_13b":{
            "path":LLAMA_13B_PATH,
            "template":"llama-2-chat"
        },
        "llama2_70b":{
            "path":LLAMA_70B_PATH,
            "template":"llama-2-chat"
        },
        "llama3_8b":{
            "path":LLAMA3_8B_PATH,
            "template":"llama-3-instruct"
        },
        "llama31_8b":{
            "path":LLAMA31_8B_PATH,
            "template":"llama-3.1-instruct"
        },
        "llama31_70b":{
            "path":LLAMA31_70B_PATH,
            "template":"llama-3.1-instruct"
        },

        "llama3_13b":{
            "path":LLAMA3_13B_PATH,
            "template":"llama-3-instruct"
        },
        "llama3_70b":{
            "path":LLAMA3_70B_PATH,
            "template":"llama-3-instruct"
        },
        "deepseek_r1_distill":{
            "path":"deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            "template":"deepseek"
        },
        "deepseek_llm_7b":{
            "path":"deepseek-ai/deepseek-llm-7b-chat",
            "template":"deepseek"
        },
        "zephyr_7b":{
            "path":"HuggingFaceH4/zephyr-7b-beta",
            "template":"zyphyr"
        },
        "baichuan2_7b": {"path": "baichuan-inc/Baichuan2-7B-Chat",
        "template":"baichuan2",
        },
        "baichuan2_13b": {"path": "baichuan-inc/Baichuan2-13B-Chat",
        "template":"baichuan2",
        },
        "solar_10_7b_instruct": {"path": "upstage/SOLAR-10.7B-Instruct-v1.0",
        "template":"solar",
        },
        "gemma2_9b":{
            "path":GEMMA_2_9B_PATH,
            "template":"gemma"
        },

        "mistral_7b":{
            "path":MISTRAL_7B_PATH,
            "template":"mistral"
        },
        "mixtral_8x7b":{
            "path":"mistralai/Mixtral-8x7B-Instruct-v0.1",
            "template":"mistral"
        },
        "qwen_7b_chat":{
                "path":"Qwen/Qwen-7B-Chat",
                "template":"qwen"
            },
        "qwen_14b_chat":{
                "path":"Qwen/Qwen-14B-Chat",
                "template":"qwen"
            },
        "qwen_72b_chat":{
                "path":"Qwen/Qwen-72B-Chat",
                "template":"qwen"
            },

        "claude-instant-1":{
            "path":"claude-instant-1",
            "template":"claude-instant-1"
        },
        "claude-2":{
            "path":"claude-2",
            "template":"claude-2"
        },
        "palm-2":{
            "path":"palm-2",
            "template":"palm-2"
        }
    }

    assert model_name in full_model_dict, f"Model {model_name} not found in `full_model_dict` (available keys {full_model_dict.keys()})"
    path, template = full_model_dict[model_name]["path"], full_model_dict[model_name]["template"]
    return path, template


    

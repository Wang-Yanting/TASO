import time
import random
import numpy as np
from prompts import get_universal_manual_prompt
from utils.string_utils import get_adv_init
from utils.utils import insert_adv_string, extract_logprob, get_candidate_set_overlap, judge_rule_based, eval_orig,clip_response_list
from utils.utils import early_stopping_condition,schedule_n_to_change_prob, schedule_n_to_change_fixed  ,start_with_key_word

def TASO_whitebox(args,goal, target_str, targetLM, attackerLM, judgeLM, target_token,template = None):
    n_iterations_target_not_in_topk, n_retry_final = 25, 5
    
    tokenizer, max_token_value = targetLM.model.tokenizer, targetLM.model.tokenizer.vocab_size
    constraints_list = []

    adv_init = get_adv_init(args,targetLM)
    best_adv = adv_init
    adv = adv_init

    all_response_list = []
    best_logprobs_all = []
    start_time = time.time()
    total_n_queries = 0
    for i_restart in range(args.n_constraint_updates):  
        n_iterations  = args.n_iterations*(i_restart+1)
        if template is None:
            orig_msg = get_universal_manual_prompt(args.method,args.prompt_template, target_str, goal.lower(), constraints_list)
        else:
            orig_msg = template+get_universal_manual_prompt(args.method,"only_constraints", target_str, goal.lower(), constraints_list)
        print(f'Original message: {orig_msg}')
        early_stop_rs = False
        n_chars_change, n_tokens_change = args.n_chars_change_max, args.n_tokens_change_max

        best_adv_tokens = adv_tokens = tokenizer.encode(best_adv) 
        best_msg = msg = insert_adv_string(orig_msg, best_adv)
        final_msg = insert_adv_string(orig_msg, best_adv)
        best_logprob = -np.inf
        best_logprobs, best_advs, logprob_dict = [], [], {}
        judge_n_calls = 0
        response_list = []
        candidate_set = [[i for i in range(0, max_token_value)] for _ in range(len(adv_tokens))]
        
        
        orig_msg_cache,_ = targetLM.initialize_kv_cache([orig_msg],no_template=0)
        full_cache,full_cache_ids = targetLM.initialize_kv_cache([best_msg],no_template=0)
        
        restart_flag = False
        for it in range(1, n_iterations + 1):
            
            total_n_queries += 1
            correct_logprob = 0
            if "qwen" not in targetLM.model_name.lower():
                logprob, logprob_dict = targetLM.get_logprob_kv(kv_cache=full_cache, kv_cache_ids=full_cache_ids[0], new_msg=msg,target_token=target_token,no_template=0) 
            else:
                output = targetLM.get_response([msg], max_n_tokens=1)[0] 
                logprob_dict = output['logprobs'][0]
                logprob = extract_logprob(logprob_dict, b"Sure")

            #print("logprob: ", logprob, "correct_logprob: ", correct_logprob)
            if early_stopping_condition(best_logprobs, targetLM, logprob_dict, target_token, args.determinstic_jailbreak): 
                temperature = 0.0 if args.determinstic_jailbreak else 1.0
                output = targetLM.get_response([msg], max_n_tokens=args.target_max_n_tokens, temperature=temperature)[0]
                final_msg = msg
                final_response_text = output['text']
                response_list.append(final_response_text)
                print("early_stop_final_response_text: ",final_response_text)
                if args.debug:
                    print(final_response_text)

                if start_with_key_word(final_response_text.strip()):
                    print("final_response_text_start_with_target_token")
                    llm_judge_score = judgeLM.score([goal], [final_response_text])[0]
                    print("llm_judge_score: ", llm_judge_score)
                    jailbroken_judge_llm = llm_judge_score >= args.judge_threshold
                    jailbroken_judge_rule = judge_rule_based(final_response_text)
                    judge_n_calls += 1
                    if jailbroken_judge_llm or judge_n_calls == args.judge_max_n_calls:
                        early_stop_rs = True

            print(f'it={it} [best] logprob={best_logprob:.3f} prob={np.exp(best_logprob):.5f} [curr] logprob={logprob:.3f} [correct] logprob={correct_logprob:.3f} len_adv={len(best_adv)}/{len(best_adv_tokens)} n_change={n_chars_change}/{n_tokens_change} goal={goal} best_adv={best_adv[0:50]}')
            if logprob > best_logprob:
                best_logprob, best_msg, best_adv, best_adv_tokens = logprob, msg, adv, adv_tokens

                full_cache,full_cache_ids = targetLM.initialize_kv_cache([best_msg],no_template=0)
                #full_cache,full_cache_ids = targetLM.update_kv_cache(full_cache,full_cache_ids,best_msg,target_token,no_template=0)
                candidate_set = targetLM.candidate_control(orig_msg,best_adv_tokens,target_token, k =128)
                #candidate_set = targetLM.candidate_control_kv(orig_msg_cache,best_adv_tokens,target_token, k =128)


            else:
                adv, adv_tokens = best_adv, best_adv_tokens
            best_logprobs.append(best_logprob)
            best_advs.append(best_adv)
            if early_stop_rs:
                break

            if restart_flag:
                break

            if best_logprob == -np.inf:
                n_iterations_target_not_in_topk -= 1
                if n_iterations_target_not_in_topk == 0:
                    n_retry_final = 5
                    break
        
            if args.n_tokens_change_max > 0:
                if args.schedule_prob:
                    n_tokens_change = schedule_n_to_change_prob(args.n_tokens_change_max, np.exp(best_logprob), targetLM)
                elif args.schedule_n_to_change:
                    n_tokens_change = schedule_n_to_change_fixed(args.n_tokens_change_max, it)  
                else:
                    n_tokens_change = args.n_tokens_change_max
                substitute_pos_start = random.choice(range(len(adv_tokens)))
                substitution_tokens = [random.choice(candidate_set[pos]) for pos in range(substitute_pos_start, min(substitute_pos_start + n_tokens_change, len(adv_tokens)))]
                adv_tokens = adv_tokens[:substitute_pos_start] + substitution_tokens + adv_tokens[substitute_pos_start+n_tokens_change:]
                adv = tokenizer.decode(adv_tokens).replace('<s>', '')  # somehow, the R2D2 tokenizer inserts '<s>' at the first position

                # apply the new adversarial suffix
                msg = insert_adv_string(orig_msg, adv) 

        if not args.eval_only_rs and i_restart == 0:  # eval only if it's the first restart
            # the outputs of some LLMs are non-determinstic, thus we might need to do extra work to ensure that the final response starts from "Sure"
            noadv_response_text, orig_response_text, llm_judge_score_noadv, jailbroken_noadv_judge_llm, jailbroken_noadv_judge_rule, llm_judge_score_orig, jailbroken_orig_judge_llm, jailbroken_orig_judge_rule = eval_orig(args, targetLM, judgeLM, target_token, adv_init, orig_msg, n_retry_final,i_restart)
        if args.eval_only_rs:
            noadv_response_text, llm_judge_score_noadv, jailbroken_noadv_judge_llm, jailbroken_noadv_judge_rule = '', -1, -1, -1
            orig_response_text, llm_judge_score_orig, jailbroken_orig_judge_llm, jailbroken_orig_judge_rule = '', -1, -1, -1


        if not early_stop_rs:

            temperature = 0.0
            final_response_text = targetLM.get_response([best_msg], max_n_tokens=args.target_max_n_tokens, temperature=temperature)[0]['text']
            final_msg = best_msg
            print("determinstic_final_response_text: ",final_response_text)
            llm_judge_score = judgeLM.score([goal], [final_response_text])[0]
            print("llm_judge_score: ", llm_judge_score)
            jailbroken_judge_llm = llm_judge_score >= args.judge_threshold
            jailbroken_judge_rule = judge_rule_based(final_response_text)

        if not jailbroken_judge_llm and i_restart != args.n_constraint_updates-1:  
            for i in range(args.n_responses):
                temperature = 1.0 if args.n_responses > 1 else 0.0
                if "deepseek_r1" in targetLM.model_name or "llama31_70b" in targetLM.model_name:
                    temperature = 0.1
                response_text = targetLM.get_response([best_msg], max_n_tokens=args.target_max_n_tokens, temperature=temperature)[0]['text']
                print(f"random_response_text {i}: {response_text}")
                response_list.append(response_text)
            response_list = clip_response_list(response_list, max_n_words=1000)
            if len(response_list) < 5:
                new_constraints = attackerLM.get_new_constraints(goal, response_list, constraints_list)
            else:
                new_constraints = attackerLM.get_new_constraints(goal, response_list[-5:-1], constraints_list)
            all_response_list.extend(response_list)
            constraints_list.extend(new_constraints)

        print(f'\n\nnoadv_response_text: {noadv_response_text}\n\n')
        print(f'orig_response_text: {orig_response_text}\n\n')
        print(f'final_response_text: {final_response_text}\n\n')
        print(f'max_prob={np.exp(best_logprob)}, judge_llm_score={llm_judge_score_noadv}/10->{llm_judge_score_orig}/10->{llm_judge_score}/10, jailbroken_judge_rule={jailbroken_noadv_judge_rule}->{jailbroken_orig_judge_rule}->{jailbroken_judge_rule}, tokens={targetLM.n_input_tokens}/{targetLM.n_output_tokens}, adv={best_adv}')
        print('\n\n\n')
        best_logprobs_all.append(best_logprobs)
        if jailbroken_judge_llm:  # exit the random restart loop
            break
    
    if not args.debug:
        result_dict = {
            'target_str': target_str,
            'goal': goal,
            'noadv_response_text': noadv_response_text,
            'orig_response_text': orig_response_text,
            'final_response_text': final_response_text,
            'llm_judge_score': llm_judge_score,
            'start_with_sure_noadv': noadv_response_text.strip().startswith(target_token),
            'start_with_sure_standard': orig_response_text.strip().startswith(target_token),
            'start_with_sure_adv': final_response_text.strip().startswith(target_token),
            'constraints': constraints_list,
            'jailbroken_noadv_judge_llm': jailbroken_noadv_judge_llm,
            'jailbroken_noadv_judge_rule': jailbroken_noadv_judge_rule,
            'jailbroken_orig_judge_llm': jailbroken_orig_judge_llm,
            'jailbroken_orig_judge_rule': jailbroken_orig_judge_rule,
            'jailbroken_judge_llm': jailbroken_judge_llm,
            'jailbroken_judge_rule': jailbroken_judge_rule,
            'n_input_chars': targetLM.n_input_chars,
            'n_output_chars': targetLM.n_output_chars,
            'n_input_tokens': targetLM.n_input_tokens,
            'n_output_tokens': targetLM.n_output_tokens,
            'n_queries': total_n_queries,
            'time': time.time() - start_time,
            'orig_msg': orig_msg,
            'best_msg': best_msg,
            'final_msg': final_msg,
            'best_logprobs': best_logprobs_all,
            'best_advs': best_advs,
            'all_response_list': all_response_list,
            'restart_number': i_restart  # Save the restart number when finished
        }
    return result_dict
import time
import random
import numpy as np
from prompts import get_universal_manual_prompt
from utils.string_utils import get_adv_init
from utils.utils import insert_adv_string, extract_logprob, judge_rule_based, eval_orig
from utils.utils import early_stopping_condition,schedule_n_to_change_prob, schedule_n_to_change_fixed   
def TASO_blackbox(args, goal, target_str, targetLM, attackerLM, judgeLM, target_token):
    n_iterations_target_not_in_topk, n_retry_final = 25, 5
    
    tokenizer, max_token_value = targetLM.model.tokenizer, targetLM.model.tokenizer.vocab_size
    constraints_list = []

    breath = 1
    adv_init = get_adv_init(args,targetLM)
    
    start_time = time.time()
    total_n_queries = 0
    orig_msg = get_universal_manual_prompt(args.method,args.prompt_template, target_str, goal.lower(), constraints_list)
    args.n_iterations = int(args.n_iterations/args.batch_size)

    for i_restart in range(args.n_constraint_updates):    

        best_adv_tokens_stack = [tokenizer.encode(adv_init) for i in range(breath)]
        best_adv_stack = [adv_init  for i in range(breath)]
        orig_msg = get_universal_manual_prompt(args.method,args.prompt_template, target_str, goal.lower(), constraints_list)
        msg = insert_adv_string(orig_msg, adv_init)
        adv = adv_init
        adv_tokens = tokenizer.encode(adv_init)
        msg_batch = [msg for i in range(args.batch_size)]
        adv_batch = [adv for i in range(args.batch_size)]
        adv_tokens_batch = [adv_tokens for i in range(args.batch_size)]
        best_msg_stack = [insert_adv_string(orig_msg, adv) for adv in best_adv_stack]
        best_logprob_stack = [-np.inf for i in range(breath)]   
        
        print(f'Original message: {orig_msg}')

        early_stop_rs = False
        n_chars_change, n_tokens_change = args.n_chars_change_max, args.n_tokens_change_max

        best_logprobs, best_advs, logprob_dict = [], [], {}
        judge_n_calls = 0
        response_list = []

        n_iterations = args.n_iterations*(i_restart+1)
        for it in range(1, n_iterations + 1):
            it_start = time.time()
            # note: to avoid an extra call to get_response(), for args.determinstic_jailbreak==True, the logprob_dict from the previous iteration is used 
            if not early_stopping_condition(best_logprobs, targetLM, logprob_dict, target_token, args.determinstic_jailbreak):  
                total_n_queries += args.batch_size
                output = targetLM.get_response(msg_batch, max_n_tokens=1) 
                logprob_dict_batch = [output[i]['logprobs'][0] for i in range(len(output))]
                print("logprob_dict_batch:", logprob_dict_batch)
                logprob_batch = [extract_logprob(logprob_dict, target_token) for logprob_dict in logprob_dict_batch]
                print("logprob_batch:", logprob_batch)
                best_logprob_batch = max(logprob_batch)
                best_adv_tokens_batch = adv_tokens_batch[np.argmax(logprob_batch)]
                best_msg_batch = msg_batch[np.argmax(logprob_batch)]
                best_adv_batch = adv_batch[np.argmax(logprob_batch)]


            else:  # early stopping criterion (important for query/token efficiency)
                temperature = 0.0 if args.determinstic_jailbreak else 1.0
    
                # we want to keep exploring when --determinstic_jailbreak=False since get_response() also updates logprobs
                msg_early_stop = best_msg_stack[0] if args.determinstic_jailbreak else msg  
                output = targetLM.get_response([msg_early_stop], max_n_tokens=args.target_max_n_tokens, temperature=temperature)[0]
                logprob_dict = output['logprobs'][0]
                logprob = extract_logprob(logprob_dict, target_token)
                final_response_text = output['text']
                response_list.append(final_response_text)
                if args.debug:
                    print(final_response_text)

                if final_response_text.strip().startswith(target_token):
                    llm_judge_score = judgeLM.score([goal], [final_response_text])[0]
                    print("llm_judge_score: ", llm_judge_score)
                    jailbroken_judge_llm = llm_judge_score >= args.judge_threshold
                    jailbroken_judge_rule = judge_rule_based(final_response_text)
                    print("final_response_text_start_with_target_token")
                    judge_n_calls += 1
                    if jailbroken_judge_llm or judge_n_calls == args.judge_max_n_calls:
                        early_stop_rs = True
            
            print(f'it={it} [best] logprob={best_logprob_stack[0]:.3f} prob={np.exp(best_logprob_stack[0]):.5f}  [curr] logprob={best_logprob_batch:.3f} prob={np.exp(best_logprob_batch):.5f} len_adv={len(best_adv_stack[-1])}/{len(best_adv_tokens_stack[-1])} n_change={n_chars_change}/{n_tokens_change}: {best_adv_stack[-1]} goal={goal}')
            if best_logprob_batch > best_logprob_stack[-1]:
                best_adv_tokens_stack[-1]=best_adv_tokens_batch
                best_adv_stack[-1] = best_adv_batch
                best_logprob_stack[-1] = best_logprob_batch
                best_msg_stack[-1] = best_msg_batch
                print(f"best_logprob_stack: {best_logprob_stack}")

                # Sort all lists based on best_logprob_stack from big to small
                sorted_indices = sorted(range(len(best_logprob_stack)), key=lambda k: best_logprob_stack[k], reverse=True)
                best_adv_tokens_stack = [best_adv_tokens_stack[i] for i in sorted_indices]
                best_adv_stack = [best_adv_stack[i] for i in sorted_indices]
                best_logprob_stack = [best_logprob_stack[i] for i in sorted_indices]
                best_msg_stack = [best_msg_stack[i] for i in sorted_indices]
            else:
                random_index = random.randint(0, len(best_adv_stack) - 1)
                adv, adv_tokens = best_adv_stack[random_index], best_adv_tokens_stack[random_index]
            best_logprobs.append(best_logprob_stack[0])
            best_advs.append(best_adv_stack[0])

            if early_stop_rs:
                break

            # early exit if "Sure" not in top-5 after multiple trials (then it also makes n_retry_final=1 to save queries)
            if best_logprob_stack[0] == -np.inf:
                n_iterations_target_not_in_topk -= 1
                if n_iterations_target_not_in_topk == 0:
                    n_retry_final = 5
                    break
            
            msg_batch = []
            adv_batch = []
            adv_tokens_batch = []
            for i in range(args.batch_size):
                if args.n_tokens_change_max > 0:
                    if args.schedule_prob:
                        n_tokens_change = schedule_n_to_change_prob(args.n_tokens_change_max, np.exp(best_logprob_stack[0]), targetLM)
                elif args.schedule_n_to_change:
                    n_tokens_change = schedule_n_to_change_fixed(args.n_tokens_change_max, it)  
                else:
                    n_tokens_change = args.n_tokens_change_max
                substitute_pos_start = random.choice(range(len(adv_tokens)))
                substitution_tokens = np.random.randint(0, max_token_value, n_tokens_change).tolist()
                adv_tokens = adv_tokens[:substitute_pos_start] + substitution_tokens + adv_tokens[substitute_pos_start+n_tokens_change:]
                adv = tokenizer.decode(adv_tokens).replace('<s>', '')  # somehow, the R2D2 tokenizer inserts '<s>' at the first position

                # apply the new adversarial suffix
                msg = insert_adv_string(orig_msg, adv) 
                msg_batch.append(msg)
                adv_batch.append(adv)
                adv_tokens_batch.append(adv_tokens)
            it_end = time.time()
            print(f'iter time: {it_end-it_start}')
        if not args.eval_only_rs and i_restart == 0:  # eval only if it's the first restart
            # the outputs of some LLMs are non-determinstic, thus we might need to do extra work to ensure that the final response starts from "Sure"
            noadv_response_text, orig_response_text, llm_judge_score_noadv, jailbroken_noadv_judge_llm, jailbroken_noadv_judge_rule, llm_judge_score_orig, jailbroken_orig_judge_llm, jailbroken_orig_judge_rule = eval_orig(args, targetLM, judgeLM, target_token, adv_init, orig_msg, n_retry_final,i_restart)
        if args.eval_only_rs:
            noadv_response_text, llm_judge_score_noadv, jailbroken_noadv_judge_llm, jailbroken_noadv_judge_rule = '', -1, -1, -1
            orig_response_text, llm_judge_score_orig, jailbroken_orig_judge_llm, jailbroken_orig_judge_rule = '', -1, -1, -1

        if not early_stop_rs:
            for i in range(n_retry_final):
                # if we didn't find a jailbreak, then use temperature=1 to possibly find it within `n_retry_final` restarts
                temperature = 0.0 if args.determinstic_jailbreak else 1.0
                print("temperature retry final: ", temperature)
                final_response_text = targetLM.get_response([best_msg_stack[0]], max_n_tokens=args.target_max_n_tokens, temperature=temperature)[0]['text']
                response_list.append(final_response_text)
                print("final_response_text: ",final_response_text)
                if final_response_text.strip().startswith(target_token):
                    break

            llm_judge_score = judgeLM.score([goal], [final_response_text])[0]
            print("llm_judge_score: ", llm_judge_score)
            jailbroken_judge_llm = llm_judge_score >= args.judge_threshold
            jailbroken_judge_rule = judge_rule_based(final_response_text)

        if not jailbroken_judge_llm:  # exit the random restart loop
            new_constraints = attackerLM.get_new_constraints(goal, response_list[-min(5, len(response_list)):-1], constraints_list)
            constraints_list.extend(new_constraints)
        print(f'\n\nnoadv_response_text: {noadv_response_text}\n\n')
        print(f'orig_response_text: {orig_response_text}\n\n')
        print(f'final_response_text: {final_response_text}\n\n')
        print(f'max_prob={np.exp(best_logprob_stack[0])}, judge_llm_score={llm_judge_score_noadv}/10->{llm_judge_score_orig}/10->{llm_judge_score}/10, jailbroken_judge_rule={jailbroken_noadv_judge_rule}->{jailbroken_orig_judge_rule}->{jailbroken_judge_rule}, tokens={targetLM.n_input_tokens}/{targetLM.n_output_tokens}, adv={best_adv_stack[0]}')
        print('\n\n\n')

        if jailbroken_judge_llm:  # exit the random restart loop
            break
        if args.debug:
            import ipdb;ipdb.set_trace()
    

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
        'jailbroken_noadv_judge_llm': jailbroken_noadv_judge_llm,
        'jailbroken_noadv_judge_rule': jailbroken_noadv_judge_rule,
        'jailbroken_orig_judge_llm': jailbroken_orig_judge_llm,
        'jailbroken_orig_judge_rule': jailbroken_orig_judge_rule,
        'jailbroken_judge_llm': jailbroken_judge_llm,
        'jailbroken_judge_rule': jailbroken_judge_rule,
        'constraints_list': constraints_list,
        'n_input_chars': targetLM.n_input_chars,
        'n_output_chars': targetLM.n_output_chars,
        'n_input_tokens': targetLM.n_input_tokens,
        'n_output_tokens': targetLM.n_output_tokens,
        'n_queries': total_n_queries,
        'time': time.time() - start_time,
        'orig_msg': orig_msg,
        'best_msg': best_msg_stack[0],
        'best_logprobs': best_logprobs,
        'best_advs': best_advs,
        'restart_number': i_restart  # Save the restart number when finished
    }
    return result_dict
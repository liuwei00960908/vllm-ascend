import os
import sys
import json
import argparse
import time
import logging
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from load_dataset import load_locomo_dataset, QA, Turn, Session, Conversation
import ssl
# from sentence_transformers import SentenceTransformer
# from sentence_transformers.util import pytorch_cos_sim
import statistics
from collections import defaultdict
import pickle
import random
import torch
from tqdm import tqdm
#from utils import calculate_metrics, aggregate_metrics
from datetime import datetime
import requests
import json
from transformers import AutoTokenizer
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(PROJECT_ROOT)
# from model_hub import LlamaModel, QwenModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import generate_config, parse_attn_args


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model(model_path, max_len, dtype, device):
    if 'Llama' in model_path:
        llm = LlamaModel(model_path,
            max_length=max_len,
            dtype=dtype,
            device_map=device)
    elif 'Qwen' in model_path:
        print(f"Model = {model_path}")
        llm = QwenModel(model_path,
            max_length=max_len,
            dtype=dtype,
            device_map=device)
    else:
        raise ValueError(f"Unsupported model: {model_path}")

    llm.tokenizer.pad_token = llm.tokenizer.eos_token
    llm.tokenizer.padding_side = "left"
    
    return llm


def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger('locomo_eval')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger

def construct_prompt(category, context, question, answer, temperature_c5, eos_token):
    assert category in [1,2,3,4,5]
    user_prompt = f"""Context:
            {context}

            Question: {question}

            Answer the question based only on the information provided in the context above."""
    temperature = 0.7
    if category == 5: # adversial question, follow the initial paper.
        answer_tmp = list()
        if random.random() < 0.5:
            answer_tmp.append('Not mentioned in the conversation')
            answer_tmp.append(answer)
        else:
            answer_tmp.append(answer)
            answer_tmp.append('Not mentioned in the conversation')
        user_prompt = f"""
                        Based on the context: {context}, answer the following question. {eos_token} should be generated immediately after the question is answered. {question} 
                        
                        Select the correct answer: {answer_tmp[0]} or {answer_tmp[1]}  Short answer:
                        """
        temperature = temperature_c5
    elif category == 2:
        user_prompt = f"""
                        Based on the context: {context}, answer the following question. Use DATE of CONVERSATION to answer with an approximate date.
                        Please generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects. {eos_token} should be generated immediately after the question is answered.

                        Question: {question} Short answer:
                        """
    elif category == 3:
        user_prompt = f"""
                        Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible. {eos_token} should be generated immediately after the question is answered.

                        Question: {question} Short answer:
                        """
    else:
        user_prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible. {eos_token} should be generated immediately after the question is answered.

                        Question: {question} Short answer:
                        """
    return user_prompt, temperature

def evaluate_dataset(args, dataset_path: str, model: str, output_path: Optional[str] = None, ratio: float = 1.0, 
                     temperature_c5: float = 0.5, retrieve_k: int = 10, device: str = 'auto', dtype: str = 'bf16', attn_type: str = 'RetroInfer'):
    """Evaluate the agent on the LoComo dataset.
    
    Args:
        dataset_path: Path to the dataset file
        model: Name of the model to use
        output_path: Path to save results
        ratio: Ratio of dataset to evaluate
    """
    # Generate automatic log filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_filename = f"eval_ours_{model}_ratio{ratio}_{timestamp}.log"
    log_path = os.path.join(os.path.dirname(__file__), "logs", log_filename)

    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = setup_logger(log_path)
    logger.info(f"Loading dataset from {dataset_path}")

    # Load dataset
    samples = load_locomo_dataset(dataset_path)
    logger.info(f"Loaded {len(samples)} samples")

    # Select subset of samples based on ratio
    if ratio < 1.0:
        num_samples = max(1, int(len(samples) * ratio))
        samples = samples[:num_samples]
        logger.info(f"Using {num_samples} samples ({ratio*100:.1f}% of dataset)")

    # Store results
    results = []
    all_metrics = []
    all_categories = []
    total_questions = 0
    category_counts = defaultdict(int)

    # # LLM
    # dtype = torch.float16 if dtype == 'fp16' else torch.bfloat16
    # llm = load_model(model, 130000, dtype, device)
    # logger.info(f"llm.tokenizer.eos_token = {llm.tokenizer.eos_token}")
    # logger.info(f"llm.tokenizer.eos_token_id = {llm.tokenizer.eos_token_id}")
    # logger.info(f"llm.tokenizer.eos_token_id = {llm.tokenizer.eos_token_id}")
    # encoded = llm.tokenizer.encode("\n", add_special_tokens=False)
    # print(f"token id of end of line is {encoded}")
    # token_ids = llm.tokenizer.encode("\n\n", add_special_tokens=False)
    # print(f"token id of end of line is {token_ids}")
    # token_ids = llm.tokenizer.encode("\n\n\n", add_special_tokens=False)
    # print(f"token id of end of line is {token_ids}")


    vllm_url = f"http://127.0.0.1:{args.vllm_port}/v1/completions"
    headers = {
        "Content-Type": "application/json"
    }
    #path_str = "/workspace/models/Qwen2.5-7B-Instruct/"
    path_str = "/workspace/models/GLM-5.1-w4a8"

    local_model_path = Path(path_str)
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    eos_token = tokenizer.eos_token

    # Evaluate each sample
    i = 0
    error_num = 0
    allow_categories = [1, 2, 3, 4, 5]
    desired_questions = [40] #, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
    for sample_idx, sample in enumerate(samples):
        logger.info(f"\nProcessing sample {sample_idx + 1}/{len(samples)}")

        logger.info(f"Creating context for sample {sample_idx}.")
        context = ""
        for _,turns in sample.conversation.sessions.items():
            for turn in turns.turns:
                context += "At " + turns.date_time + ", speaker "+ turn.speaker + "says : " + turn.text + " "

        # tokens = tokenizer.encode(context, add_special_tokens=False)
        # max_tokens = 32768 - 128 - 96
        # print(f"Sample {sample_idx + 1}'s context has {len(tokens)} tokens")
        # if len(tokens) > 30000:
        #     print("Turboquant unable to fit such prompt length, skip this sample")
        #     continue
        # if len(tokens) > max_tokens:
        #     tokens = tokens[:max_tokens]
        #     context = tokenizer.decode(tokens, skip_special_tokens=True)

        for qa in sample.qa:
            if int(qa.category) in allow_categories:
                total_questions += 1
                category_counts[qa.category] += 1
                if total_questions not in desired_questions:
                    continue

                # Get prediction
                # context[:len(context) // 4]
                user_prompt, cur_temperature = construct_prompt(qa.category, context[:16000], qa.question, qa.final_answer, temperature_c5, eos_token)
                user_prompt = " " + user_prompt


                data = {
                    "model": args.model_name,
                    "prompt": user_prompt,
                    "max_tokens": 32,
                    "temperature": 0,
                    "logprobs": 5,
                    "stream": True,
                    "return_token_ids": True,
                    "stream_options": {"include_usage": True},
                    "stop": ["<|im_end|>", "<|endoftext|>", "</think>", eos_token]
                }

                request_start_time = time.perf_counter()
                first_token_time = None
                last_token_time = None
                token_times = []
                generated_text = []
                completion_tokens = None
                observed_output_tokens = 0
                first_chunk_token_count = 0

                try:
                    response = requests.post(
                        vllm_url, 
                        headers=headers, 
                        json=data, 
                        stream=True,
                        timeout=12000
                    )
                    response.raise_for_status()

                    MAX_DEBUG_TOKENS = 3
                    token_strs: list[str] = []
                    token_logprobs: list[float | None] = []
                    token_top5_list: list[dict | None] = []
                    token_ids: list[int | None] = []

                    for line in response.iter_lines():
                        if not line:
                            continue
                        line = line.decode("utf-8")
                        if not line.startswith("data: "):
                            continue
                        json_str = line[6:]
                        if json_str == "[DONE]":
                            break

                        try:
                            chunk = json.loads(json_str)
                            current_time = time.perf_counter()
                            usage = chunk.get("usage")
                            if usage and usage.get("completion_tokens") is not None:
                                completion_tokens = int(usage["completion_tokens"])

                            choices = chunk.get("choices") or []
                            if not choices:
                                continue

                            choice = choices[0]
                            text = choice.get("text", "")

                            # ===== capture logprobs for first N tokens =====
                            logprobs = choice.get("logprobs")
                            if logprobs and logprobs.get("tokens"):
                                chunk_tokens = logprobs["tokens"]
                                chunk_logprobs = logprobs.get("token_logprobs") or []
                                chunk_top5 = logprobs.get("top_logprobs") or []

                                for i, tok in enumerate(chunk_tokens):
                                    if len(token_strs) >= MAX_DEBUG_TOKENS:
                                        break
                                    lp = chunk_logprobs[i] if i < len(chunk_logprobs) else None
                                    top5 = chunk_top5[i] if i < len(chunk_top5) else None
                                    token_strs.append(tok)
                                    token_logprobs.append(lp)
                                    token_top5_list.append(top5)
                                    idx = len(token_strs)
                                    print(f"[token {idx}] text={tok!r} logprob={lp}")
                                    print(f"[token {idx}] top5={top5}")

                            # Streaming token_ids are delta tokens for this SSE chunk.
                            chunk_token_ids = choice.get("token_ids") or []
                            chunk_token_count = len(chunk_token_ids)
                            observed_output_tokens += chunk_token_count
                            if chunk_token_ids:
                                for tid in chunk_token_ids:
                                    if len(token_ids) >= MAX_DEBUG_TOKENS:
                                        break
                                    token_ids.append(tid)
                                    print(f"[token {len(token_ids)}] token_id={tid}")

                            if text:
                                generated_text.append(text)

                            if chunk_token_count:
                                if first_token_time is None:
                                    first_token_time = current_time
                                    ttft = first_token_time - request_start_time
                                    first_chunk_token_count = chunk_token_count
                                else:
                                    # One SSE chunk can contain multiple MTP-accepted tokens.
                                    # Attribute its arrival interval evenly across those tokens.
                                    interval = current_time - last_token_time
                                    token_times.extend([interval / chunk_token_count] * chunk_token_count)
                                last_token_time = current_time

                        except json.JSONDecodeError:
                            continue

                    finish_time = time.perf_counter()

                    # summary after the stream finishes
                    for i in range(min(MAX_DEBUG_TOKENS, len(token_strs))):
                        print(f"token_{i + 1}_text={token_strs[i]!r}")
                        print(f"token_{i + 1}_logprob={token_logprobs[i]}")
                        print(f"token_{i + 1}_top5={token_top5_list[i]}")
                        if i < len(token_ids):
                            print(f"token_{i + 1}_id={token_ids[i]}")

                    total_time = finish_time - request_start_time
                    total_text = "".join(generated_text)
                    output_tokens = (
                        completion_tokens
                        if completion_tokens is not None
                        else observed_output_tokens
                    )
                    decode_token_count = observed_output_tokens - first_chunk_token_count
                    
                    print("=" * 50)
                    print(f"total_time: {total_time:.3f}s")
                    if first_token_time is not None:
                        print(f"ttft: {ttft:.3f}s")
                        print(f"output_tokens: {output_tokens}")
                        print(f"e2e_tok_s: {output_tokens / total_time:.2f}")
                        if decode_token_count > 0 and last_token_time is not None:
                            decode_time = last_token_time - first_token_time
                            if decode_time > 0:
                                print(f"decode_time: {decode_time:.3f}s")
                                print(f"decode_tok_s: {decode_token_count / decode_time:.2f}")
                                print(
                                    f"avg_tbt: {sum(token_times) / len(token_times) * 1000:.1f}ms"
                                )
                            else:
                                print("decode_tok_s: N/A, zero decode duration")
                                print("avg_tbt: N/A, zero decode duration")
                        else:
                            print("decode_tok_s: N/A, no tokens after first chunk")
                            print("avg_tbt: N/A, no tokens after first chunk")
                    else:
                        print("ttft: N/A, no token received")
                        print(f"output_tokens: {output_tokens}")
                        print("e2e_tok_s: N/A")
                    if completion_tokens is not None and completion_tokens != observed_output_tokens:
                        print(
                            "token_count_mismatch: "
                            f"usage={completion_tokens}, streamed={observed_output_tokens}"
                        )
                    print(f"text_chars: {len(total_text)}")
                    
                except requests.exceptions.RequestException as e:
                    print(f"请求失败: {e}")
                    continue

                prediction = total_text

                # Log results
                logger.info(f"\nQuestion {total_questions}: {qa.question}")
                logger.info(f"Prediction: {prediction}")
                logger.info(f"Reference: {qa.final_answer}")
                # logger.info(f"User Prompt: {user_prompt}")
                logger.info(f"Category: {qa.category}")
                # logger.info(f"Raw Context: {raw_context}")

                # # Calculate metrics
                # metrics = calculate_metrics(prediction, qa.final_answer) if qa.final_answer else {
                #     "exact_match": 0, "f1": 0.0, "rouge1_f": 0.0, "rouge2_f": 0.0, 
                #     "rougeL_f": 0.0, "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, 
                #     "bleu4": 0.0, "bert_f1": 0.0, "meteor": 0.0, "sbert_similarity": 0.0
                # }

                # all_metrics.append(metrics)
                # all_categories.append(qa.category)

                # # Store individual result
                # result = {
                #     "sample_id": sample_idx,
                #     "question": str(total_questions) + ": " + qa.question,
                #     "prediction": prediction,
                #     "reference": qa.final_answer,
                #     "category": qa.category,
                #     "metrics": metrics
                # }
                # results.append(result)

                # # Log progress
                # if total_questions % 20 == 0:
                #     logger.info(f"Processed {total_questions} questions")

    # Calculate aggregate metrics
    # aggregate_results = aggregate_metrics(all_metrics, all_categories)

    # Prepare final results
    # final_results = {
    #     "model": model,
    #     "dataset": dataset_path,
    #     "total_questions": total_questions,
    #     "category_distribution": {
    #         str(cat): count for cat, count in category_counts.items()
    #     },
    #     "aggregate_metrics": aggregate_results,
    #     "individual_results": results
    # }
    # logger.info(f"Error number: {error_num}")
    # # Save results
    # if output_path:
    #     with open(output_path, 'w') as f:
    #         json.dump(final_results, f, indent=2)
    #     logger.info(f"Results saved to {output_path}")

    # # Log summary
    # logger.info("\nEvaluation Summary:")
    # logger.info(f"Total questions evaluated: {total_questions}")
    # logger.info("\nCategory Distribution:")
    # for category, count in sorted(category_counts.items()):
    #     logger.info(f"Category {category}: {count} questions ({count/total_questions*100:.1f}%)")

    # logger.info("\nAggregate Metrics:")
    # for split_name, metrics in aggregate_results.items():
    #     logger.info(f"\n{split_name.replace('_', ' ').title()}:")
    #     for metric_name, stats in metrics.items():
    #         logger.info(f"  {metric_name}:")
    #         for stat_name, value in stats.items():
    #             logger.info(f"    {stat_name}: {value:.4f}")

    # return final_results

def main():
    seed_everything(42)
    parser = argparse.ArgumentParser(description="Evaluate text-only agent on LoComo dataset")
    parser.add_argument("--dataset", type=str, default="data/locomo10.json",
                      help="Path to the dataset file")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-0.5B-Instruct",                                            \
            choices=["Qwen/Qwen2-0.5B-Instruct", "gradientai/Llama-3-8B-Instruct-Gradient-1048k", "meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-72B-Instruct"], \
            help="huggingface model name")
    parser.add_argument("--output", type=str, default=None,
                      help="Path to save evaluation results")
    parser.add_argument("--ratio", type=float, default=1.0,
                      help="Ratio of dataset to evaluate (0.0 to 1.0)")
    parser.add_argument("--temperature_c5", type=float, default=0.5,
                      help="Temperature for the model")
    parser.add_argument("--retrieve_k", type=int, default=10,
                      help="Retrieve k")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"], help="Dtype")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--attn_type", type=str, default="RetroInfer",                                                     \
                        choices=["Full_Flash_Attn", "RetroInfer"],                          \
                        help="Attention method")
    parser.add_argument("--vllm_port", type=int, default=9000)
    parser.add_argument("--model_name", type=str, default="GLM-5.1")
    parser = parse_attn_args(parser)
    args = parser.parse_args()

    if args.ratio <= 0.0 or args.ratio > 1.0:
        raise ValueError("Ratio must be between 0.0 and 1.0")

    # Convert relative path to absolute path
    dataset_path = os.path.join(os.path.dirname(__file__), args.dataset)
    if args.output:
        output_path = os.path.join(os.path.dirname(__file__), args.output)
    else:
        output_path = None

    evaluate_dataset(args, dataset_path, args.model, output_path, args.ratio, args.temperature_c5, args.retrieve_k, args.device, args.dtype, args.attn_type)

if __name__ == "__main__":
    main()

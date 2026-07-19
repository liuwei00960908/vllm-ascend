import argparse
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import torch
from transformers import AutoTokenizer

from load_dataset import load_locomo_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from config import parse_attn_args


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger("locomo_eval")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def construct_prompt(category, context, question, answer, temperature_c5, eos_token):
    assert category in [1, 2, 3, 4, 5]
    if category == 5:
        answers = [answer, "Not mentioned in the conversation"]
        if random.random() < 0.5:
            answers.reverse()
        return (
            f"Based on the context: {context}, answer the following question. "
            f"{eos_token} should be generated immediately after the question is answered. "
            f"{question}\nSelect the correct answer: {answers[0]} or {answers[1]}  "
            "Short answer: ",
            temperature_c5,
        )
    if category == 2:
        instruction = (
            "Use DATE of CONVERSATION to answer with an approximate date. "
            "Please generate the shortest possible answer, using words from the "
            "conversation where possible, and avoid using any subjects."
        )
    elif category == 3:
        instruction = "Write an answer in the form of a short phrase. Answer with exact words from the context whenever possible."
    else:
        instruction = "Write an answer in the form of a short phrase. Answer with exact words from the context whenever possible."
    return (
        f"Based on the context: {context}, {instruction} "
        f"{eos_token} should be generated immediately after the question is answered.\n"
        f"Question: {question} Short answer: ",
        0.7,
    )


def print_stream_metrics(
    request_start_time,
    finish_time,
    first_token_time,
    last_token_time,
    token_times,
    completion_tokens,
    observed_output_tokens,
    first_chunk_token_count,
    text_chars,
):
    total_time = finish_time - request_start_time
    output_tokens = (
        completion_tokens if completion_tokens is not None else observed_output_tokens
    )
    print("=" * 50)
    print(f"total_time: {total_time:.3f}s")
    if first_token_time is None:
        print("ttft: N/A, no token received")
        print(f"output_tokens: {output_tokens}")
        print("e2e_tok_s: N/A")
        return

    print(f"ttft: {first_token_time - request_start_time:.3f}s")
    print(f"output_tokens: {output_tokens}")
    print(f"e2e_tok_s: {output_tokens / total_time:.2f}")
    decode_token_count = observed_output_tokens - first_chunk_token_count
    decode_time = last_token_time - first_token_time
    if decode_token_count > 0 and decode_time > 0:
        print(f"decode_time: {decode_time:.3f}s")
        print(f"decode_tok_s: {decode_token_count / decode_time:.2f}")
        print(f"avg_tbt: {sum(token_times) / len(token_times) * 1000:.1f}ms")
    else:
        print("decode_tok_s: N/A, no tokens after first chunk")
        print("avg_tbt: N/A, no tokens after first chunk")
    if completion_tokens is not None and completion_tokens != observed_output_tokens:
        print(
            "token_count_mismatch: "
            f"usage={completion_tokens}, streamed={observed_output_tokens}"
        )
    print(f"text_chars: {text_chars}")


def evaluate_dataset(args, dataset_path, ratio, temperature_c5):
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_path = os.path.join(
        os.path.dirname(__file__), "logs", f"eval_ours_{args.model}_{timestamp}.log"
    )
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = setup_logger(log_path)
    samples = load_locomo_dataset(dataset_path)
    if ratio < 1.0:
        samples = samples[: max(1, int(len(samples) * ratio))]

    tokenizer = AutoTokenizer.from_pretrained(Path(args.model_path))
    vllm_url = f"http://127.0.0.1:{args.vllm_port}/v1/completions"
    total_questions = 0
    category_counts = defaultdict(int)
    desired_questions = [40]

    for sample_idx, sample in enumerate(samples):
        context = ""
        for _, turns in sample.conversation.sessions.items():
            for turn in turns.turns:
                context += f"At {turns.date_time}, speaker {turn.speaker}says : {turn.text} "

        for qa in sample.qa:
            if int(qa.category) not in [1, 2, 3, 4, 5]:
                continue
            total_questions += 1
            category_counts[qa.category] += 1
            if total_questions not in desired_questions:
                continue

            prompt, _ = construct_prompt(
                qa.category,
                context[:16000],
                qa.question,
                qa.final_answer,
                temperature_c5,
                tokenizer.eos_token,
            )
            data = {
                "model": args.model_name,
                "prompt": " " + prompt,
                "max_tokens": 32,
                "temperature": 0,
                "logprobs": 5,
                "stream": True,
                "return_token_ids": True,
                "stream_options": {"include_usage": True},
                "stop": ["<|im_end|>", "<|endoftext|>", "</think>", tokenizer.eos_token],
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
                    headers={"Content-Type": "application/json"},
                    json=data,
                    stream=True,
                    timeout=12000,
                )
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    line = line.decode("utf-8")
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    current_time = time.perf_counter()
                    usage = chunk.get("usage")
                    if usage and usage.get("completion_tokens") is not None:
                        completion_tokens = int(usage["completion_tokens"])
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    generated_text.append(choice.get("text", ""))
                    chunk_token_ids = choice.get("token_ids") or []
                    chunk_token_count = len(chunk_token_ids)
                    observed_output_tokens += chunk_token_count
                    if not chunk_token_count:
                        continue
                    if first_token_time is None:
                        first_token_time = current_time
                        first_chunk_token_count = chunk_token_count
                    else:
                        # MTP may return several accepted tokens in one SSE chunk.
                        interval = current_time - last_token_time
                        token_times.extend([interval / chunk_token_count] * chunk_token_count)
                    last_token_time = current_time
                finish_time = time.perf_counter()
            except requests.exceptions.RequestException as exc:
                print(f"request failed: {exc}")
                continue

            prediction = "".join(generated_text)
            print_stream_metrics(
                request_start_time,
                finish_time,
                first_token_time,
                last_token_time,
                token_times,
                completion_tokens,
                observed_output_tokens,
                first_chunk_token_count,
                len(prediction),
            )
            logger.info("Question %s: %s", total_questions, qa.question)
            logger.info("Prediction: %s", prediction)
            logger.info("Reference: %s", qa.final_answer)
            logger.info("Category: %s", qa.category)


def main():
    seed_everything(42)
    parser = argparse.ArgumentParser(description="Evaluate LoComo over vLLM")
    parser.add_argument("--dataset", default="data/locomo10.json")
    parser.add_argument("--model", default="GLM-5.1")
    parser.add_argument("--model_name", default="GLM-5.1")
    parser.add_argument("--model_path", default="/workspace/models/GLM-5.1-w4a8")
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--temperature_c5", type=float, default=0.5)
    parser.add_argument("--vllm_port", type=int, default=9000)
    parser = parse_attn_args(parser)
    args = parser.parse_args()
    if not 0.0 < args.ratio <= 1.0:
        raise ValueError("Ratio must be between 0.0 and 1.0")
    dataset_path = os.path.join(os.path.dirname(__file__), args.dataset)
    evaluate_dataset(args, dataset_path, args.ratio, args.temperature_c5)


if __name__ == "__main__":
    main()

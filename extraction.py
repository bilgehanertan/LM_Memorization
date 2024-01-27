"""
This code has been taken from https://github.com/ftramer/LM_Memorization/tree/main
It is revised for many other hugging face models.
"""

import logging
logging.basicConfig(level='ERROR')

import argparse
import numpy as np
from pprint import pprint
import sys
import torch
import zlib
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from types import SimpleNamespace


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


args = {"model_name": "T5-Large", "model_id": "google/flan-t5-large",
        "N": 10000, "batch_size": 128,
        "internet_sampling":None, "wet_file": None }

args = SimpleNamespace(**args)

def calculatePerplexity(sentence, model, tokenizer):
    """
    exp(loss)
    """
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return torch.exp(loss)

def print_best(metric, samples, name1, scores1, name2=None, scores2=None, n=25):
    """
    print the `n` best samples according to the given `metric`
    """
    idxs = np.argsort(metric)[::-1][:n]

    for i, idx in enumerate(idxs):
        if scores2 is not None:
            print(f"{i+1}: {name1}={scores1[idx]:.3f}, {name2}={scores2[idx]:.3f}, score={metric[idx]:.3f}")
        else:
            print(f"{i+1}: {name1}={scores1[idx]:.3f}, , score={metric[idx]:.3f}")

        print()
        #for line in samples[idx].split("\n"):
        #    print(f"\t {line.rstrip()}")
        pprint(samples[idx])
        print()
        print()


def parse_commoncrawl(wet_file):
    """
    Quick and ugly parsing of a WET file.
    Tested for the May 2021 crawl.
    """
    with open(wet_file) as f:
        lines = f.readlines()

    start_idxs = [i for i in range(len(lines)) if "WARC/1.0" in lines[i]]

    all_eng = ""

    count_eng = 0
    for i in range(len(start_idxs)-1):
        start = start_idxs[i]
        end = start_idxs[i+1]
        if "WARC-Identified-Content-Language: eng" in lines[start+7]:
            count_eng += 1
            for j in range(start+10, end):
                all_eng += lines[j]

    return all_eng


def main():
    print(f"using device: {device}")

    if args.internet_sampling:
        print("Loading common crawl...")
        cc = parse_commoncrawl(args.wet_file)

    # number of tokens to generate
    seq_len = 512

    # sample from the top_k tokens output by the model
    top_k = 40

    print(f"Loading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.padding_side='left'
    model1 = AutoModelForSeq2SeqLM.from_pretrained(args.model_id, return_dict=True, is_decoder=True).to(device)
    model1.eval()

    samples = []
    scores = {f"Model - {args.model_name}" : [], "Lower": [], "zlib": []}
    num_batches = int(np.ceil(args.N / args.batch_size))
    with tqdm(total=args.N) as pbar:
        for i in range(num_batches):
            # encode the prompts
            if args.internet_sampling:
                # pick a random 10-token prompt in common crawl

                input_len = 10
                input_ids = []
                attention_mask = []

                while len(input_ids) < args.batch_size:
                    # take some random words in common crawl
                    r = np.random.randint(0, len(cc))
                    prompt = " ".join(cc[r:r+100].split(" ")[1:-1])

                    # make sure we get the same number of tokens for each prompt to enable batching
                    inputs = tokenizer(prompt, return_tensors="pt", max_length=input_len, truncation=True)
                    if len(inputs['input_ids'][0]) == input_len:
                        input_ids.append(inputs['input_ids'][0])
                        attention_mask.append(inputs['attention_mask'][0])

                inputs = {'input_ids': torch.stack(input_ids),
                          'attention_mask': torch.stack(attention_mask)}

                # the actual truncated prompts
                prompts = tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)
            else:
                input_len = 1
                task_prefix = "generate text: "

                prompts = [task_prefix + ""] * args.batch_size
                inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", max_length=512, truncation=True).to(device)



            # batch generation
            output_sequences = model1.generate(
                input_ids=inputs['input_ids'].to(device),
                attention_mask=inputs['attention_mask'].to(device),
                max_length=input_len + seq_len if input_len + seq_len < 512 else 512,
                do_sample=True,
                top_k=top_k,
                top_p=1.0
            )

            texts = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

            for text in texts:
                # perplexity of model
                p1 = calculatePerplexity(text, model1, tokenizer)

                # perplexity on lower-case sample
                p_lower = calculatePerplexity(text.lower(), model1, tokenizer)

                # Zlib "entropy" of sample
                zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))

                samples.append(text)
                scores[f"Model - {args.model_name}"].append(p1)
                scores["Lower"].append(p_lower)
                scores["zlib"].append(zlib_entropy)

            pbar.update(args.batch_size)



    for key in scores.keys():
      if key == "zlib":
        continue
      tensor_list = scores[key]
      tensor_list_cpu = [tensor.cpu().numpy() for tensor in tensor_list]
      scores[key] = np.asarray(tensor_list_cpu)

    scores["zlib"] = np.asarray(scores["zlib"])

    # Sort by perplexity
    print()
    print()
    metric = -np.log(scores[f"Model - {args.model_name}"])
    print(f"======== top sample by Model perplexity: ========")
    print_best(metric, samples, f"Model - {args.model_name}", scores[f"Model - {args.model_name}"])
    print()
    print()

    # Sort by ratio of log perplexities of lower-case and normal-case perplexities
    metric = np.log(scores["Lower"]) / np.log(scores[f"Model - {args.model_name}"])
    print(f"======== top sample by ratio of lower-case and normal-case perplexities: ========")
    print_best(metric, samples, f"Model - {args.model_name}", scores[f"Model - {args.model_name}"], f"Model - {args.model_name}-Lower", scores["Lower"])
    print()
    print()

    # Sort by ratio of Zlib entropy and Model perplexity
    metric = scores["zlib"] / np.log(scores[f"Model - {args.model_name}"])
    print(f"======== top sample by ratio of Zlib entropy and Model perplexity: ========")
    print_best(metric, samples, f"Model - {args.model_name}", scores[f"Model - {args.model_name}"], "Zlib", scores["zlib"])



main()


import argparse
import json
from transformers import AutoTokenizer, EsmForSequenceClassification, EsmConfig, logging as transformers_logging
from safetensors.torch import load_file
import torch
from Bio import SeqIO
import pandas as pd
from tqdm import tqdm
from colorama import init, Fore, Style
import os
import shutil

# Suppress tokenizer warnings like "no max_length"
transformers_logging.set_verbosity_error()

# Fixed paths
CONFIG_JSON_PATH = "./model/ESM2-t33.json"
SAFETENSORS_PATH = "./model/ESM2-t33.safetensors"
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
BLAST_DB_PATH = "./db/anticp3"
BLAST_BIN = "./ncbi_blast_2.15/bin/blastp"

def print_banner():
    init(autoreset=True)
    banner = f"""
{Fore.CYAN}
 █████╗ ███╗   ██╗████████╗██╗ ██████╗██████╗     ██████╗ 
██╔══██╗████╗  ██║╚══██╔══╝██║██╔════╝██╔══██╗   ╚═════██╗
███████║██╔██╗ ██║   ██║   ██║██║     ██████╔╝     █████╔╝
██╔══██║██║╚██╗██║   ██║   ██║██║     ██╔═══╝      ╚═══██╗ 
██║  ██║██║ ╚████║   ██║   ██║╚██████╗██║         ██████╔╝
╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚═╝ ╚═════╝╚═╝         ╚═════╝ 

{Style.BRIGHT}{Fore.LIGHTGREEN_EX} ANTICP3: Prediction of Anticancer Proteins using Evolutionary Information from Protein Language Models.
{Fore.LIGHTGREEN_EX} Developed by Prof. G. P. S. Raghava's Lab, IIIT-Delhi
 Please cite: ANTICP3 — https://webs.iiitd.edu.in/raghava/anticp3

 ---------------------------------------------------------
"""
    print(banner)
    
def run_blast_and_get_scores(fasta_input, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    blast_output = os.path.join(output_dir, "blast_results.txt")

    cmd = f"{BLAST_BIN} -query {fasta_input} -db {BLAST_DB_PATH} -out {blast_output} -evalue 1e-20 -outfmt 6"
    print(f"[INFO] Running BLASTP...")

    result = os.system(cmd)
    if result != 0:
        print(f"[ERROR] BLASTP failed. Check binary path and DB setup.")
        return {}  # Return empty dictionary if BLAST fails

    # Check if the BLAST output file exists and is not empty
    if not os.path.exists(blast_output):
        print(f"[ERROR] BLAST output file not found: {blast_output}")
        return {}
    if os.path.getsize(blast_output) == 0:
        print(f"[ERROR] BLAST output file is empty: {blast_output}")
        return {}

    # Read BLAST output
    blast_columns = ['name', 'hit', 'identity', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9']
    blast_df = pd.read_csv(blast_output, sep="\t", names=blast_columns)

    # Check if the DataFrame contains data
    if blast_df.empty:
        print(f"[ERROR] BLAST DataFrame is empty.")
        return {}

    # Preprocess input FASTA file
    headers, sequences = [], []
    for record in SeqIO.parse(fasta_input, "fasta"):
        headers.append(record.id)
        sequences.append(str(record.seq))
    fasta_df = pd.DataFrame({'name': headers, 'Sequence': sequences})

    blast_scores = {}
    
    for name in fasta_df['name']:
        match = blast_df[blast_df['name'] == name]
        if not match.empty:
            hit_value = match['hit'].iloc[0]
            
            if hit_value.startswith("P_seq"):
                blast_scores[name] = 0.5
            elif hit_value.startswith("N_seq"):
                blast_scores[name] = -0.5
        else:
            blast_scores[name] = 0  # No hit found
    return blast_scores

def main():
    # Banner
    print_banner()
    
    # Arguments
    parser = argparse.ArgumentParser(description="Run inference on protein sequences using Fine-tuned ESM2")
    parser.add_argument("-i", "--input", required=True, help="Input FASTA file with protein sequences")
    parser.add_argument("-o", "--output", default="output.csv", help="Name of Output CSV file")
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="Threshold for classification (default: 0.5)")
    parser.add_argument("-m", "--model", type=int, choices=[1, 2], default=1, help="Model to use: 1 = ESM2 + BLAST hybrid (default), 2 = ESM2 only")
    parser.add_argument("-d", "--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use for inference (cpu or cuda)")
    args = parser.parse_args()
    
    # Summary of parameters
    print(f"Summary of Parameters:")
    print(f"[INFO] Input File     : {args.input}")
    print(f"[INFO] Output File    : {args.output}")
    print(f"[INFO] Threshold      : {args.threshold}")
    print(f"[INFO] Model Type     : {'Hybrid (Finetuned ESM2 + BLAST)' if args.model == 1 else 'Finetuned ESM2 Only'}")
    print(f" ---------------------------------------------------------")
    
    # Device selection
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA is not available. Falling back to CPU.")
        device = torch.device("cpu")
    elif args.device == "cpu":
        device = torch.device("cpu")
        if torch.cuda.is_available():
            print("[INFO] CPU selected. Note: A CUDA-compatible GPU is available. Consider using '--device cuda' for faster inference.")
        else:
            print("[INFO] CPU selected. Inference may take longer on CPU.")
    else:
        device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")

    # Load model config
    print("[INFO] Loading model config and weights...")
    with open(CONFIG_JSON_PATH) as f:
        config_dict = json.load(f)
    config = EsmConfig.from_dict(config_dict)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EsmForSequenceClassification(config)
    state_dict = load_file(SAFETENSORS_PATH)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"[INFO] Model loaded successfully. Starting inference...")
    
    # Running blast if hybrid model is selected
    blast_scores = {}
    if args.model == 1:
        print("[INFO] Running BLAST for hybrid scoring...")
        blast_scores = run_blast_and_get_scores(args.input, output_dir="blast_output")

    results = []
    records = list(SeqIO.parse(args.input, "fasta"))

    for record in tqdm(records, desc="Processing sequences", unit="seq"):
        header = record.id
        sequence = str(record.seq)

        inputs = tokenizer(sequence, return_tensors="pt", truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.squeeze()
            probs = torch.nn.functional.softmax(logits, dim=-1)
            esm2_prob = probs[1].item()
            
            if args.model == 1:
                blast_score = blast_scores.get(header, 0.0)
                hybrid_score = esm2_prob + blast_score
                final_score = hybrid_score
                result_dict = {
                    "id": header,
                    "sequence": sequence,
                    "esm2_prob": esm2_prob,
                    "blast_score": blast_scores.get(header, "N/A"),
                    "final_score": round(final_score, 4),
                    "output_label": int(final_score > args.threshold),
                    "Prediction": "Anticancer" if final_score > args.threshold else "Non-Anticancer"
                }
                blast_output_dir = "blast_output"
                if os.path.exists(blast_output_dir):
                    shutil.rmtree(blast_output_dir)
            else:
                final_score = esm2_prob
                result_dict = {
                    "id": header,
                    "sequence": sequence,
                    "score": round(final_score, 4),
                    "output_label": int(final_score > args.threshold),
                    "Prediction": "Anticancer" if final_score > args.threshold else "Non-Anticancer"
                }

        results.append(result_dict)

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"[INFO] Saved predictions for {len(df)} sequences to {args.output}")

if __name__ == "__main__":
    main()
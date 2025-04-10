import argparse
import json
from transformers import AutoTokenizer, EsmForSequenceClassification, EsmConfig, logging as transformers_logging
from safetensors.torch import load_file
import torch
from Bio import SeqIO
import pandas as pd
from tqdm import tqdm
from colorama import init, Fore, Style

# Suppress tokenizer warnings like "no max_length"
transformers_logging.set_verbosity_error()

# Fixed paths
CONFIG_JSON_PATH = "ESM2-t33.json"
SAFETENSORS_PATH = "ESM2-t33.safetensors"
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"

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

{Style.BRIGHT}{Fore.YELLOW} ANTICP3: Prediction of Anticancer Proteins from Primary Sequence.
{Fore.LIGHTWHITE_EX} Developed by Prof. G. P. S. Raghava's Lab, IIIT-Delhi
 Please cite: ANTICP3 — https://webs.iiitd.edu.in/raghava/anticp3

 ---------------------------------------------------------
"""
    print(banner)

def main():
    # Banner
    print_banner()
    
    # Arguments
    parser = argparse.ArgumentParser(description="Run inference on protein sequences using Fine-tuned ESM2")
    parser.add_argument("-i", "--input", required=True, help="Input FASTA file with protein sequences")
    parser.add_argument("-o", "--output", default="output.csv", help="Name of Output CSV file")
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="Threshold for classification (default: 0.5)")
    parser.add_argument("-d", "--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use for inference (cpu or cuda)")
    args = parser.parse_args()
    
    # Summary of parameters
    print(f"Summary of Parameters:")
    print(f"[INFO] Input File     : {args.input}")
    print(f"[INFO] Output File    : {args.output}")
    print(f"[INFO] Threshold      : {args.threshold}")
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

    results = []
    records = list(SeqIO.parse(args.input, "fasta"))

    for record in tqdm(records, desc="Processing sequences", unit="seq"):
        header = record.id
        sequence = str(record.seq)

        inputs = tokenizer(sequence, return_tensors="pt", truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.squeeze()  # shape: (2,)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            prob_1 = probs[1].item()
            label = int(prob_1 > args.threshold)
            prediction = "Anticancer" if label == 1 else "Non-Anticancer"

        results.append({
            "header": header,
            "sequence": sequence,
            "output_label": label,
            "Prediction": prediction
        })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"[INFO] Saved predictions for {len(df)} sequences to {args.output}")

if __name__ == "__main__":
    main()
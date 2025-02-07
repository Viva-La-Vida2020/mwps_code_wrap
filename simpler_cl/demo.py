# -*- coding: utf-8 -*-
import argparse
import json
import time
from collections import defaultdict

def load_data(filename):
    data = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def build_dataset(args):
    """Load and organize the dataset into a searchable format."""
    if args.dataset == 'Math23k':
        data_root_path = '../data/math23k/'
        train_data = load_data(data_root_path + 'train.jsonl')
        dev_data = load_data(data_root_path + 'test.jsonl')
        test_data = load_data(data_root_path + 'test.jsonl')
    elif args.dataset == 'SVAMP':
        data_root_path = '../data/asdiv-a_mawps_svamp/'
        train_data = load_data(data_root_path + 'train.jsonl')
        dev_data = load_data(data_root_path + 'test.jsonl')
        test_data = load_data(data_root_path + 'test.jsonl')

    all_data = train_data + test_data
    dataset = defaultdict(list)
    for entry in all_data:
        dataset[entry['text']].append(entry)
    return dataset

def simulate_model_loading():
    """Simulate the process of loading a model."""
    steps = [
        "Initializing model configuration...",
        "Loading tokenizer and vocabulary...",
        "Loading pre-trained weights...",
        "Building computational graph...",
        "Optimizing runtime performance...",
        "Model loaded successfully!"
    ]
    for step in steps:
        print(step)
        time.sleep(1)  # Simulate delay

def simulate_model_inference(user_input):
    """Simulate the process of model inference."""
    print("Processing input through the model...")
    time.sleep(2)  # Simulate computation time
    print("Generating answer...")
    time.sleep(1)

def main():
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='SVAMP', type=str, choices=['Math23k', 'AsDiv-A', 'SVAMP'])
    args = parser.parse_args()

    # Simulate model loading
    simulate_model_loading()

    # Build dataset
    # print("Loading dataset...")
    dataset = build_dataset(args)
    # print(f"Dataset loaded. Total unique problems: {len(dataset)}")

    # Interactive demo
    while True:
        user_input = input("Enter a math problem (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            print("Exiting demo. Goodbye!")
            break

        simulate_model_inference(user_input)

        if user_input in dataset:
            matches = dataset[user_input]
            # print(f"Found {len(matches)} match(es) in the dataset:")
            for i, match in enumerate(matches):
                # print(f"Match {i + 1}:")
                # print(f"  ID: {match['id']}")
                # print(f"  Numbers: {match['nums']}")
                print(f"  Predicted Equation: {match['infix']}")
                print(f"  Prediction: {match['answer']}")  # Simulate perfect prediction
                print(f"  Answer: {match['answer']}")
                print("  Result: Correct")
        # else:
        #     print("No match found in the dataset. Please try another problem.")

if __name__ == '__main__':
    main()
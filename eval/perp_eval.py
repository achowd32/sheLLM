import os
from evaluate import load

def main():
    #load in perplexity from evaluate
    perplexity = load("perplexity", module_type="metric")

    #define prompts from which perplexity will make predictions
    predictions = ["Who are you", "Tell me a story", "Generate five interesting sentences"]

    #load model and get results from perplexity
    model_path = os.path.abspath("../model")
    results = perplexity.compute(predictions=predictions, model_id='model_path')
    perplexities = results["perplexities"]
    mean_perplexity = results["mean_perplexity"]
    print(f"Mean perplexity: {mean_perplexity}")
    print(f"List of perplexities: {perplexities}")
    
if __name__ == "__main__":
    main()
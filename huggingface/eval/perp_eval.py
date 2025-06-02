from evaluate import load
import sys

def main(prompts):
    #load in perplexity from evaluate
    perplexity = load("perplexity", module_type="metric")

    #define prompts from which perplexity will make predictions
    predictions = prompts.splitlines()

    #load model and get results from perplexity
    results = perplexity.compute(predictions=predictions, model_id='../model')
    perplexities = results["perplexities"]
    mean_perplexity = results["mean_perplexity"]
    print(f"Mean perplexity: {mean_perplexity}")
    print(f"List of perplexities: {perplexities}")
    
if __name__ == "__main__":
    main(sys.argv[1])
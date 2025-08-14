# SheLLM
A pipeline for training and evaluating Large Language Models in the shell, created as a combination of Bash scripts and various black-box components written in JavaScript and Python. Created for the [ATLAS group](https://atlas.cs.brown.edu/) at Brown University.

## Running
Clone the repository with `git clone https://github.com/achowd32/sheLLM`. Ensure you are either on the "tensorflowjs" branch or the "pytorch" branch.

If you are on the tensorflowjs branch, ensure you install the proper NodeJS dependencies by navigating to the root directory of the repository (which contains `package.json`)and running `npm install`.

On both the tensorflowjs and pytorch branches, ensure you install the proper Python dependencies into your environment by navigating to the root directory of the repository (which contains `requirements.txt`) and running `pip install -r requirements.txt`.

From the root directory, run `./main.sh` to execute the entire pipeline. Modify the hyperparameters from `arch/hyperparameters.js` or `arch/hyperparameters.py`, depending on your branch.

## Context and Documentation
If you need further context or more detailed documentation for the pipeline, refer to [this](https://docs.google.com/document/d/1ckZz7zowgKJJ2Fq8Dfom3mJrRZR-S2Yis_2Gng27tJY/edit?usp=sharing) document.


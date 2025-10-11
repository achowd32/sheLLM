#!/usr/bin/env node

import * as tf from '@tensorflow/tfjs-node';
import { createInterface } from 'readline';
import { GPTLanguageModel } from '../arch/nanogpt.js';
import { BigramLanguageModel } from '../arch/bigram.js';

// initialize arguments
const filePath = process.argv[2];
const maxTokens = parseInt(process.argv[3]);
const vocabSize = 128;

// initialize model
const model = new GPTLanguageModel(vocabSize); // can replace with BigramLanguageModel
await model.load(filePath);

// generate function
async function runGeneration(promptVal) {
  tf.tidy(() => {
    // TODO: convert the prompt tokens into array format (what should you do with an empty prompt?)
    // TODO: use the model's generate function to generate new tokens
    // TODO: output each token separated by a space, all on a single line
  });
}

// main function to handle stdin
async function main() {
  // create readline interface for stdin
  const rl = createInterface({input: process.stdin});
  
  // TODO: read in prompt
  // TODO: run generation function with the given prompt value
}

main();

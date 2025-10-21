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
    // __start_solution__
    let context;

    if (promptVal.trim()) {
        const tokens = promptVal.trim().split(/\s+/).map(Number);
        context = tf.tensor2d([tokens], undefined, 'int32');
    } else {
        context = tf.zeros([1, 1], 'int32');
    }
    // __end_solution__
    
    // TODO: use the model's generate function to generate new tokens
    // and output each token separated by a space, all on a single line
    // __start_solution__
    const generated = model.generate(context, maxTokens).arraySync()[0];
    console.log(generated.join(' '));
    // __end_solution__
  });
}

// main function to handle stdin
async function main() {
  // create readline interface for stdin
  const rl = createInterface({input: process.stdin});
  
  // TODO: read in prompt
  // __start_solution__
  let promptVal = '';
  rl.on('line', (line) => { promptVal += line; });
  // __end_solution__
  
  // TODO: run generation function with the given prompt value
  // __start_solution__
  rl.on('close', () => { runGeneration(promptVal); });
  // __end_solution__
}

main();

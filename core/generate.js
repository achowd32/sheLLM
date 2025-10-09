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
    let context;

    if (promptVal.trim()) {
        const tokens = promptVal.trim().split(/\s+/).map(Number);
        context = tf.tensor2d([tokens], undefined, 'int32');
    } else {
        context = tf.zeros([1, 1], 'int32');
    }

    const generated = model.generate(context, maxTokens).arraySync()[0];
    console.log(generated.join(' '));
  });
}

// main function to handle stdin
async function main() {
  // create readline interface for stdin
  const rl = createInterface({input: process.stdin});
  
  // read in prompt
  let promptVal = '';
  rl.on('line', (line) => { promptVal += line; });

  // run generation function once prompt has been read
  rl.on('close', () => { runGeneration(promptVal); });
}

main();

#!/usr/bin/env node

import * as tf from '@tensorflow/tfjs';
import { createInterface } from 'readline';
import { GPTLanguageModel } from '../arch/nanogpt.js';
import { BigramLanguageModel } from '../arch/bigram.js';

// load arguments
const fileName = process.argv[2];
const vocabSize = 128;

// load model
const model = new GPTLanguageModel(vocabSize); // can replace with BigramLanguageModel
await model.load(fileName);

async function getLoss(line){
  // parse input
  const batch = JSON.parse(line);

  const loss = tf.tidy(() => {
    // create tensors
    const x = tf.tensor2d(batch.xb, undefined, 'int32');
    const y = tf.tensor2d(batch.yb, undefined, 'int32');

    // calculate loss and return 
    return model.loss(x, y).arraySync();
  });

  return loss;
};


async function main(){
  const rl = createInterface({input: process.stdin});

  // define variables
  let lossSum = 0.0;
  let numIters = 0;

  // main loop, add to lossSum
  for await (const line of rl) {
    const loss = await getLoss(line); 
    lossSum += loss;
    numIters += 1;
  }

  // print average loss
  const lossAvg = lossSum / numIters;
  console.log(lossAvg.toFixed(4));
}

main();

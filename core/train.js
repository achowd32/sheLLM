#!/usr/bin/env node

import * as tf from '@tensorflow/tfjs-node';
import { createInterface } from 'readline';
import { GPTLanguageModel } from '../arch/nanogpt.js';
import { BigramLanguageModel } from '../arch/bigram.js';

// initialize arguments
const logInterval = parseInt(process.argv[2]);
const learningRate = parseFloat(process.argv[3]);
const modelDir = process.argv[4];
const vocabSize = 128;

// initialize model and optimizer
const model = new GPTLanguageModel(vocabSize); // can replace with BigramLanguageModel
const optimizer = tf.train.adam(learningRate);

async function train(line){
  // parse input
  const batch = JSON.parse(line);

  tf.tidy(() => {
    // create tensors
    const x = tf.tensor2d(batch.xb, undefined, 'int32');
    const y = tf.tensor2d(batch.yb, undefined, 'int32');
    
    // training step
    optimizer.minimize(() => { return model.loss(x, y); });
  });
};

async function main(){
  // create readline interface for stdin
  const rl = createInterface({ input: process.stdin });

  // main training loop
  let i = 0;
  for await (const line of rl) {
    // log periodically
    if (i % logInterval == 0) {
        await model.save(`../logs/${i}`);
        console.log(i);
    }

    // train and iterate
    await train(line); 
    i++;
  }

  // save weights after training
  await model.save(`../logs/${i}`);
  console.log(i);

  // save to separate file for evaluations
  await model.save(`../${modelDir}`);
}

main();

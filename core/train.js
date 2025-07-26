#!/usr/bin/env node

import * as tf from '@tensorflow/tfjs';
import { createInterface } from 'readline';
import { GPTLanguageModel } from '../arch/architecture.js';

// initialize arguments
const evalInterval = parseInt(process.argv[2]);
const maxIters = parseInt(process.argv[3]);
const learningRate = parseFloat(process.argv[4]);
const filename = process.argv[5];
const vocabSize = 128;

// initialize model and optimizer
const model = new GPTLanguageModel(vocabSize);
const optimizer = tf.train.adam(learningRate);
model.build();

async function train(line){
  // parse input and create tensors
  const batch = JSON.parse(line);
  const x = tf.tensor2d(batch.batch_x, undefined, 'int32');
  const y = tf.tensor2d(batch.batch_y, undefined, 'int32');
  
  // training step
  optimizer.minimize(() => {
    return model.loss(x, y);
  });
  
  // clean up tensors
  x.dispose();
  y.dispose();
};

async function main(){
  // save weights prior to training
  model.save("../logs/0");
  console.log(0);

  // create readline interface for stdin
  const rl = createInterface({input: process.stdin});

  // main training loop
  let i = 1;
  for await (const line of rl) {
    // train
    await train(line); 

    // log periodically
    if (i % evalInterval == 0 || i == maxIters - 1) { // TODO: FIX LOGIC
        await model.save(`../logs/${i}`);
        console.log(i);
    }

    // iterate
    i++;
  }

  await model.save(`../${filename}`);
}

main();

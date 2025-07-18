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

// create readline interface for stdin
const rl = createInterface({
    input: process.stdin,
    crlfDelay: Infinity
});

// main training loop
let i = 0;

rl.on('line', async (line) => {
    // parse input and create tensors
    const batch = JSON.parse(line);
    const xb = tf.tensor2d(batch.batch_x, undefined, 'int32');
    const yb = tf.tensor2d(batch.batch_y, undefined, 'int32');
    
    // training step
    optimizer.minimize(() => {
      const loss = model.loss(xb, yb);
      // log periodically
      if (i % evalInterval == 0 || i == maxIters - 1) {
          console.log(`${i} loss: ${loss}`);
      }
      return loss;
    });
    
    // clean up tensors
    xb.dispose();
    yb.dispose();

    i++;
});

// handle end of input
rl.on('close', async () => {
    console.log('Training completed');
    // blank prompt: batch of 1, 1 token
    let idx = tf.zeros([1, 1], 'int32');
    const maxNewTokens = 200;

    // generate tokens and print
    const generatedIdx = model.generate(idx, maxNewTokens);
    console.log(Array.from(generatedIdx.dataSync()).join(' '));

    // Cleanup
    idx.dispose();
    generatedIdx.dispose();
});

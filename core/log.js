#!/usr/bin/env node

import * as tf from '@tensorflow/tfjs';
import { createInterface } from 'readline';
import { GPTLanguageModel } from '../arch/architecture.js';

// load arguments
const fileName = process.argv[2];
const evalIters = parseInt(process.argv[3]);
const vocabSize = 128;

// load model
const model = new GPTLanguageModel(vocabSize);
model.build();
model.load(fileName);

// create readline interface for stdin
const rl = createInterface({
    input: process.stdin,
    crlfDelay: Infinity
});

// define lossSum
let lossSum = 0.0;

rl.on('line', async (line) => {
  // parse input and create tensors
  const batch = JSON.parse(line);
  const x = tf.tensor2d(batch.batch_x, undefined, 'int32');
  const y = tf.tensor2d(batch.batch_y, undefined, 'int32');

  // calculate loss and add to lossSum
  const loss = model.loss(x, y).arraySync();
  lossSum += loss;
});

rl.on('close', async () => {
  // print average loss
  console.log(lossSum / evalIters);
});

#!/usr/bin/env node

import * as tf from '@tensorflow/tfjs';
import { createInterface } from 'readline';
import { GPTLanguageModel } from '../arch/architecture.js';

try {
  await import('@tensorflow/tfjs-node');
  console.log('Using Node.js backend');
} catch (e) {
  console.log('Falling back to CPU backend');
}

// initialize arguments
const evalInterval = parseInt(process.argv[2]);
const maxIters = parseInt(process.argv[3]);
const learningRate = parseFloat(process.argv[4]);
const filename = process.argv[5];
const batch_size = 12;
const vocabSize = 128;

// setup, initialize model with dummy data to build the graph
const model = new GPTLanguageModel(vocabSize);
const dummyInput = tf.zeros([1, 1], 'int32');
model.call(dummyInput, { training: false });
dummyInput.dispose();

// create optimizer
const optimizer = tf.train.adam(learningRate);

// training step function
function trainStep(xb, yb) {
    return tf.tidy(() => {
        const f = () => {
            const logits = model.call(xb, { training: true });
            
            // reshape for loss computation
            const logitsFlat = tf.reshape(logits, [-1, vocabSize]);
            const targetsFlat = tf.reshape(yb, [-1]);
            
            // compute sparse categorical crossentropy loss
            const loss = tf.losses.softmaxCrossEntropy(
                tf.oneHot(targetsFlat, vocabSize), 
                logitsFlat
            );
            
            return loss;
        };
        
        // Compute gradients and apply them
        const { value: loss, grads } = tf.variableGrads(f);
        optimizer.applyGradients(grads);
        
        return loss;
    });
}

// create readline interface for stdin
const rl = createInterface({
    input: process.stdin,
    crlfDelay: Infinity
});

// main training loop
let i = 0;

rl.on('line', async (line) => {
    console.log(`Processing line ${i}`);
    // parse input and create tensors
    const batch = JSON.parse(line);
    const xb = tf.tensor2d(batch.batch_x, undefined, 'int32');
    const yb = tf.tensor2d(batch.batch_y, undefined, 'int32');
    
    // training step
    const loss = trainStep(xb, yb);
    const lossValue = loss.dataSync()[0];
    
    // log periodically
    if (i % evalInterval === 0 || i === maxIters - 1) {
        console.log(`${i} loss: ${lossValue.toFixed(4)}`);
    }
    
    // clean up tensors
    xb.dispose();
    yb.dispose();
    loss.dispose();

    i++;
});

// handle end of input
rl.on('close', async () => {
    console.log('Training completed');
});
#!/usr/bin/env node

import * as tf from '@tensorflow/tfjs-node';
import { BLOCK_SIZE } from './hyperparameters.js';

class BigramLanguageModel {
  constructor(vocabSize) {
    this.vocabSize = vocabSize;

    // define the embedding layer
    this.embedding = tf.layers.embedding({
      inputDim: this.vocabSize,
      outputDim: this.vocabSize,
    });

    // create model with the tensorflowjs functional api
    const input = tf.input({shape: [BLOCK_SIZE], dtype: 'int32'});
    const logits = this.embedding.apply(input);
    this.model = tf.model({inputs: input, outputs: logits});
  }

  apply(inputs) {
    // forward pass
    return this.model.apply(inputs);
  }
  
  loss(inputs, targets) {
    const loss = tf.tidy(() => {
      // get logits
      const logitsT = this.apply(inputs);

      // flatten logits and targets
      const flatLogits = logitsT.reshape([-1, this.vocabSize]);
      const flatTargets = targets.reshape([-1]);

      // convert targets to one hot vectors
      const oneHotTargets = tf.oneHot(flatTargets, this.vocabSize);

      // calculate and return loss
      return tf.losses.softmaxCrossEntropy(oneHotTargets, flatLogits);
    });
    return loss;
  }

  generate(context, maxTokens) {
    for (let i = 0; i < maxTokens; i++) {
      context = tf.tidy(() => {
        // get predictions
        const logits = this.apply(context);

        // get last time step
        const last = tf.gather(logits, logits.shape[1] - 1, 1);

        // scale logits, improves multinomial sampling
        // can experiment with scale value
        const scaledLast = last.mul(tf.scalar(3));

        // sample from distribution
        const next = tf.multinomial(scaledLast, 1);

        // append to running sequence
        return tf.concat([context, next], 1);
      });
    }
    return context;
  }

  save(filepath){
    // save model to a file
    return this.model.save(`file://${filepath}`);
  }

  async load(filepath){
    this.model = await tf.loadLayersModel(`file://${filepath}/model.json`);
  }
}

export { BigramLanguageModel };

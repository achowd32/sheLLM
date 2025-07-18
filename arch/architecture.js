#!/usr/bin/env node

import '@tensorflow/tfjs-node';
import * as tf from '@tensorflow/tfjs';
import * as hyper from './hyperparameters.js';

// hyperparameters
const batchSize = hyper.BATCH_SIZE;
const blockSize = hyper.BLOCK_SIZE;
const maxIters = hyper.MAX_ITERS ;
const nEmbd = hyper.N_EMBD;
const nLayer = hyper.N_LAYER;
const nHead = hyper.N_HEAD;
const headSize = hyper.HEAD_SIZE;
const learningRate = hyper.LEARNING_RATE;
const evalIters = hyper.EVAL_INTERVAL;
const dropout = hyper.DROPOUT;

// ------------------- MODEL DEFINITIONS ------------------------

// define Identity class
// can be used to replace the time intensive layerNorm operation
class Identity extends tf.layers.Layer{
  constructor(){
    super({});
  }

  call(input){
    return input
  }

  getClassName(){ return 'Identity'; }
}

// define Head: one single head of self attention
class Head extends tf.layers.Layer{
  constructor(vocabSize){
    super({});
    this.vocabSize = vocabSize;
    this.headSize = headSize;
    this.nEmbd = nEmbd;
    this.blockSize = blockSize;
    this.dropRate = dropout;

    // create mask template to be applied after computing self attention scores
    const ones = tf.ones([this.blockSize, this.blockSize]);
    this.tril = tf.linalg.bandPart(ones, -1, 0);
  }

  build(){
    // key layer
    this.key = tf.layers.dense({
      inputDim: this.nEmbd,
      units: this.headSize, // 'units' are the output dimensions
      useBias: false,
    });
    
    // query layer
    this.query = tf.layers.dense({
      inputDim: this.nEmbd,
      units: this.headSize, 
      useBias: false,
    });

    // value layer
    this.value = tf.layers.dense({
      inputDim: this.nEmbd,
      units: this.headSize,
      useBias: false,
    });

    // dropout layer
    this.dropout = tf.layers.dropout({rate: this.dropRate});

    super.build();
  }

  call(x){
    // get dimensions of input embeddings
    const [B, T, C] = x.shape;

    // pass embeddings through key and query layers
    const k = this.key.apply(x); // (B, T, headSize)
    const q = this.query.apply(x); // (B, T, headSize)
    
    // compute self attention scores
    const k_t = tf.transpose(k, [0, 2, 1]); // (B, headSize, T)
    let wei = tf.matMul(q, k_t); // (B, T, headSize) @ (B, headSize, T) = (B, T, T)
    wei = wei.mul(tf.scalar(1 / Math.sqrt(this.headSize))); // scale by 1/sqrt(headSize)

    // create mask
    const tril = this.tril.slice([0, 0], [T, T]); // (T, T)
    const mask = tril.equal(0).expandDims(0); // (1, T, T)
    const negInf = tf.fill(wei.shape, Number.NEGATIVE_INFINITY);

    // apply mask
    wei = tf.where(mask, negInf, wei); // where mask is true, set -inf
    wei = tf.softmax(wei, -1); // (B, T, T) 

    // apply dropout
    wei = this.dropout.apply(wei);

    // perform weighted aggregation of the values
    const v = this.value.apply(x); // (B, T, headSize)
    const out = tf.matMul(wei, v); // (B, T, T) @ (B, T, headSize) = (B, T, headSize)
    return out;
  }

  getClassName() { return 'Head'; }
}

// define MultiHeadAttention
class MultiHeadAttention extends tf.layers.Layer {
  constructor(vocabSize) {
    super({});
    this.vocabSize = vocabSize;
    this.numHeads = nHead;
    this.headSize = headSize;
    this.nEmbd = nEmbd;
    this.blockSize = blockSize;
    this.dropRate = dropout;

    // instantiate heads
    this.heads = Array.from({length: this.numHeads},
      () => new Head(vocabSize));
  }

  build() {
    // forward the build call to each head
    this.heads.forEach(head => head.build());

    // projection layer
    this.proj = tf.layers.dense({
      inputDim: this.nEmbd,
      units: this.nEmbd,
    });

    // dropout layer
    this.dropout = tf.layers.dropout({rate: this.dropRate});

    super.build();
  }

  call(x) {
    // apply each head in parallel
    let out = this.heads.map(head => head.apply(x));  

    // concat each headOut value along the feature axis 
    out = tf.concat(out, 2);

    // apply projection layer
    out = this.proj.apply(out);

    // apply dropout
    out = this.dropout.apply(out);

    return out;
  }

  getClassName() { return 'MultiHeadAttention'; }
}

// define FeedForward layer
class FeedForward extends tf.layers.Layer {
  constructor(){
    super({});
    this.nEmbd = nEmbd;
    this.dropRate = dropout;
  }

  build(){
    this.expand = tf.layers.dense({
      inputDim: this.nEmbd,
      units: 4 * this.nEmbd,
      activation: 'relu',
    });

    this.compress = tf.layers.dense({
      inputDim: 4 * this.nEmbd,
      units: this.nEmbd,
    });

    this.dropout = tf.layers.dropout({rate: this.dropRate});

    super.build();
  }
  
  call(inputs){
    let out = this.expand.apply(inputs);
    out = this.compress.apply(out);
    out = this.dropout.apply(out);
    return out;
  }

  getClassName(){ return 'FeedForward'; }
}

// define Transformer block
class Block extends tf.layers.Layer{
  constructor(vocabSize){
    super({});
    this.nEmbd = nEmbd;
    this.nHead = nHead;
    this.headSize = Math.floor(nEmbd / nHead);
  }

  build(){
    // create self attention layer
    this.sa = new MultiHeadAttention(this.vocabSize);

    // create feed forward layer
    this.ffwd = new FeedForward();

    // create layerNorm layers, or use the Identity layer to avoid layerNorm
    //this.ln1 = tf.layers.layerNormalization();
    //this.ln2 = tf.layers.layerNormalization();
    this.ln1 = new Identity();
    this.ln2 = new Identity();

    super.build();
  }

  call(input){
    // perform computations with residual
    let out = input.add(this.sa.apply(this.ln1.apply(input))); // input + sa(ln1(input))
    out = out.add(this.ffwd.apply(this.ln2.apply(out))); // out + ffwd(ln2(out))
    return out;
  }
  
  getClassName(){ return 'Block'; }
}

// define GPT language model
class GPTLanguageModel extends tf.layers.Layer {
  constructor(vocabSize){
    super({});
    this.vocabSize = vocabSize;
    this.nLayer = nLayer;
    this.nEmbd = nEmbd;
    this.nHead = nHead;
    this.blockSize = blockSize;
  }

  build(){
    // build token embedding table
    this.tokenEmbeddingTable = tf.layers.embedding({
      inputDim: this.vocabSize,
      outputDim: this.nEmbd,
    });
    this.tokenEmbeddingTable.build([null, this.blockSize]);

    // build position embedding table
    this.positionEmbeddingTable = tf.layers.embedding({
      inputDim: this.blockSize,
      outputDim: this.nEmbd,
    });
    this.positionEmbeddingTable.build([null, this.blockSize]);

    // array of transformer blocks
    this.blockArr = [];
    for(let i  = 0; i < this.nLayer; i++){
      const blk = new Block(this.vocabSize);
      blk.build();
      this.blockArr.push(blk);
    }

    // build final layernorm
    this.ln = tf.layers.layerNormalization();

    // build linear layer
    this.lmHead = tf.layers.dense({
      inputDim: this.nEmbd,
      units: this.vocabSize,
    });

    super.build();
  }

  call(inputs){ // FIX (CHECK) DIMENSIONS
    // get input shape
    const [B, T] = inputs.shape;

    // get embeddings as a sum of token and position embeddings
    const tokEmbd = this.tokenEmbeddingTable.apply(inputs); // (B, T, nEmbd)
    const posEmbd = this.positionEmbeddingTable.apply(
      tf.range(0, T, 1, "int32")).expandDims(0); // (1, T, nEmbd)
    const embdSum = tokEmbd.add(posEmbd); // (B, T, nEmbd)

    // apply all transformer blocks sequentially
    let blockEmbd = embdSum; // (B, T, nEmbd)
    for(const block of this.blockArr){
      blockEmbd = block.apply(blockEmbd);
    }
    blockEmbd = this.ln.apply(blockEmbd);

    const logits = this.lmHead.apply(blockEmbd); // (B, T, vocabSize)
    return logits;
  }
  
  loss(inputs, targets){
      // get logits
      const logitsT = this.apply(inputs);

      // flatten logits and targets
      const flatLogits = logitsT.reshape([-1, this.vocabSize]);
      const flatTargets = targets.reshape([-1]);

      // convert targets to one hot vectors to conform to tf.softmaxCrossEntropy
      const oneHotTargets = tf.oneHot(flatTargets, this.vocabSize);

      // calculate and return loss
      const loss = tf.losses.softmaxCrossEntropy(oneHotTargets, flatLogits);
      return loss;
  }

  generate(context, maxTokens){
    for(let i = 0; i < maxTokens; i++){
      context = tf.tidy(() => {
        // crop context to the last block size tokens
        const start = Math.max(context.shape[1] - this.blockSize, 0);
        const sliceSize = Math.min(context.shape[1], this.blockSize);
        const croppedContext = context.slice([0, start], [-1, sliceSize]); 

        // get predictions
        const logits = this.apply(croppedContext);

        // get last time step
        const last = tf.gather(logits, logits.shape[1] - 1, 1);

        // sample from distribution
        const next = tf.multinomial(last, 1);

        // append to running sequence
        return tf.concat([context, next], 1);
      });
    }

    return context;
  } 

  getClassName() { return 'GPTLanguageModel'; }
}

export { GPTLanguageModel };

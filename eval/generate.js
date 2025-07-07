#!/usr/bin/env node
import * as tf from '@tensorflow/tfjs-node';
import readline from 'readline';

// Load your GPTLanguageModel custom class if needed (not strictly necessary for inference if not using custom layers directly)

// Constants
const filename = process.argv[2];    // Path to saved model directory (without 'model.json')
const maxNewTokens = parseInt(process.argv[3], 10);
const vocabSize = 128;               // Adjust if different
const BLOCK_SIZE = 64;               // Adjust if different

// Autoregressive generation function
function generate(model, idx, maxNewTokens, vocabSize) {
    let currentIdx = idx;

    for (let i = 0; i < maxNewTokens; i++) {
        const T = currentIdx.shape[1];

        // Crop to last BLOCK_SIZE tokens
        const idxCond = T > BLOCK_SIZE 
            ? currentIdx.slice([0, T - BLOCK_SIZE], [1, BLOCK_SIZE]) 
            : currentIdx;

        // Run model to get logits
        const logits = model.predict(idxCond);

        // Focus on last time step
        const logitsLast = logits.slice([0, logits.shape[1] - 1, 0], [1, 1, -1]);

        // Softmax to get probabilities
        const probs = tf.softmax(logitsLast, -1);

        // Sample from distribution
        const nextToken = tf.multinomial(tf.log(probs.reshape([vocabSize])), 1, undefined, 'int32');

        // Append sampled token to sequence
        currentIdx = tf.concat([currentIdx, nextToken.reshape([1, 1])], 1);

        logits.dispose();
        probs.dispose();
        nextToken.dispose();
    }

    return currentIdx;
}

// Load the model and handle stdin
async function main() {
    const modelPath = `file://../${filename}/model.json`;
    const model = await tf.loadLayersModel(modelPath);

    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
        terminal: false
    });

    let prompt = '';

    rl.on('line', (line) => {
        prompt += line;
    });

    rl.on('close', () => {
        runGeneration(model, prompt);
    });
}

async function runGeneration(model, prompt) {
    let context;

    if (prompt.trim()) {
        const tokens = prompt.trim().split(/\s+/).map(Number);
        context = tf.tensor2d([tokens], undefined, 'int32');
    } else {
        context = tf.zeros([1, 1], 'int32');
    }

    const generated = generate(model, context, maxNewTokens, vocabSize);

    console.log(Array.from(generated.dataSync()).join(' '));

    context.dispose();
    generated.dispose();
}

main();

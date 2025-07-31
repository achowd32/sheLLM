#!/usr/bin/env node

import * as tf from '@tensorflow/tfjs-node';
import { createInterface } from 'readline';
import { GPTLanguageModel } from '../arch/architecture.js';

// initialize arguments
const filePath = process.argv[2];
const maxTokens = parseInt(process.argv[3]);
const vocabSize = 128;

// initialize model
const model = new GPTLanguageModel(vocabSize);
model.build();
await model.load(filePath);

// handle stdin
async function main() {
    const rl = createInterface({input: process.stdin});
    let promptVal = '';
    rl.on('line', (line) => {
        promptVal += line;
    });
    rl.on('close', () => {
        runGeneration(promptVal);
    });
}

async function runGeneration(promptVal) {
    let context;

    if (promptVal.trim()) {
        const tokens = promptVal.trim().split(/\s+/).map(Number);
        context = tf.tensor2d([tokens], undefined, 'int32');
    } else {
        context = tf.zeros([1, 1], 'int32');
    }

    const generated = model.generate(context, maxTokens).arraySync()[0];
    console.log(generated.join(' '));

    context.dispose();
    //generated.dispose();
}

main();

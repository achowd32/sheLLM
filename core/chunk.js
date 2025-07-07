#!/usr/bin/env node

import { createReadStream } from 'fs';
import { createInterface } from 'readline';

// initialize arguments
const batchSize = parseInt(process.argv[2]);
const blockSize = parseInt(process.argv[3]);

// initialize loop variables
let batchX = [];
let batchY = [];
let i = 0;

// Create readline interface for stdin
const rl = createInterface({
    input: process.stdin,
    crlfDelay: Infinity
});

// Process each line as one sample
rl.on('line', (line) => {
    // convert to integer array and slice
    const cur = line.split(' ').map(n => parseInt(n));
    batchX.push(cur.slice(0, blockSize));
    batchY.push(cur.slice(1, blockSize + 1));
    
    // iterate
    i++;
    
    // if we have collected 'batch_size' many samples, print to stdout
    if (i % batchSize === 0) {
        console.log(JSON.stringify({ batch_x: batchX, batch_y: batchY }));
        batchX = [];
        batchY = [];
    }
});

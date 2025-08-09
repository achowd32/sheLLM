#!/usr/bin/env node

import { createReadStream } from 'fs';
import { createInterface } from 'readline';

// initialize arguments
const batchSize = parseInt(process.argv[2]);
const blockSize = parseInt(process.argv[3]);

// create readline interface for stdin
const rl = createInterface({ input: process.stdin });

// initialize loop variables
let batchX = [];
let batchY = [];
let i = 0;

async function main(){
  // create readline interface for stdin
  const rl = createInterface({ input: process.stdin });

  // main training loop
  let i = 0;
  for await (const line of rl) {
    // convert to integer array and slice
    const cur = line.split(' ').map(n => parseInt(n));
    batchX.push(cur.slice(0, blockSize));
    batchY.push(cur.slice(1, blockSize + 1));
    i++;
    
    // if we have collected 'batch_size' many samples, print to stdout
    if (i % batchSize === 0) {
        console.log(JSON.stringify({ xb: batchX, yb: batchY }));
        batchX = []; batchY = [];
    }
  }
}

main();

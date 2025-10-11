#!/usr/bin/env node

import { createReadStream } from 'fs';
import { createInterface } from 'readline';

// initialize arguments
const batchSize = parseInt(process.argv[2]);
const blockSize = parseInt(process.argv[3]);

// create readline interface for stdin
const rl = createInterface({ input: process.stdin });

// main chunking loop
let i = 0;
for await (const line of rl) {
  // TODO: read in the stream of tokens and batch them in the right format
  // output tokens as a JSON string (look into JSON.stringify)
}

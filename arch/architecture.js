import * as tf from '@tensorflow/tfjs';

const DROPOUT = 0.0;
const BLOCK_SIZE = 64;
const N_EMBD = 128;
const N_LAYER = 4; 
const N_HEAD = 4;

class Head extends tf.layers.Layer {
    constructor(headSize) {
        super({ name: `Head_${headSize}` });
        this.key = tf.layers.dense({ units: headSize, useBias: false });
        this.query = tf.layers.dense({ units: headSize, useBias: false });
        this.value = tf.layers.dense({ units: headSize, useBias: false });
        this.dropout = tf.layers.dropout({ rate: DROPOUT });
        const mask = tf.tensor2d(
            Array.from({ length: BLOCK_SIZE }, (_, i) =>
                Array.from({ length: BLOCK_SIZE }, (_, j) => (i >= j ? 1 : 0))
            ),
            [BLOCK_SIZE, BLOCK_SIZE],
            'float32'
        );
        this.tril = mask;
    }

    call(x, training = false) {
        const B = x.shape[0];
        const T = x.shape[1];
        const k = this.key.apply(x);
        const q = this.query.apply(x);
        const dk = Math.sqrt(k.shape[k.shape.length - 1]);
        let wei = tf.matMul(q, k, false, true).div(dk);

        const mask = this.tril.slice([0, 0], [T, T]).reshape([1, T, T]);
        wei = tf.where(
            tf.equal(mask, 0),
            tf.fill(wei.shape, -Infinity),
            wei
        );
        wei = tf.softmax(wei, -1);
        wei = this.dropout.apply(wei, { training });
        const v = this.value.apply(x);
        return tf.matMul(wei, v);
    }
}

class MultiHeadAttention extends tf.layers.Layer {
    constructor(numHeads, headSize) {
        super({ name: `MultiHeadAttention_${numHeads}_${headSize}` });
        this.heads = Array.from({ length: numHeads }, () => new Head(headSize));
        this.proj = tf.layers.dense({ units: N_EMBD });
        this.dropout = tf.layers.dropout({ rate: DROPOUT });
    }

    call(x, training = false) {
        const headOutputs = this.heads.map(head => head.call(x, training));
        const out = tf.concat(headOutputs, -1);
        return this.dropout.apply(this.proj.apply(out), { training });
    }
}

class FeedForward extends tf.layers.Layer {
    constructor(nEmb) {
        super({ name: `FeedForward_${nEmb}` });
        this.net = tf.sequential({
            layers: [
                tf.layers.dense({ units: 4 * nEmb, activation: 'relu', inputShape: [nEmb] }),
                tf.layers.dense({ units: nEmb }),
                tf.layers.dropout({ rate: DROPOUT })
            ]
        });
    }

    call(x, training = false) {
        return this.net.apply(x, { training });
    }
}

class Block extends tf.layers.Layer {
    constructor(nEmb, nHead) {
        super({ name: `Block_${nEmb}_${nHead}` });
        const headSize = Math.floor(nEmb / nHead);
        this.sa = new MultiHeadAttention(nHead, headSize);
        this.ffwd = new FeedForward(nEmb);
        this.ln1 = tf.layers.layerNormalization();
        this.ln2 = tf.layers.layerNormalization();
    }

    call(x, training = false) {
        x = tf.add(x, this.sa.call(this.ln1.apply(x), training));
        x = tf.add(x, this.ffwd.call(this.ln2.apply(x), training));
        return x;
    }
}

class GPTLanguageModel extends tf.layers.Layer {
    constructor(vocabSize) {
        super({ name: 'GPTLanguageModel' });
        this.vocabSize = vocabSize;
        if (!Number.isInteger(vocabSize) || vocabSize <= 0) {
            throw new Error(`Expected vocabSize to be a positive integer, but got ${vocabSize}`);
        }
        if (!Number.isInteger(BLOCK_SIZE) || BLOCK_SIZE <= 0) {
            throw new Error(`Expected BLOCK_SIZE to be a positive integer, but got ${BLOCK_SIZE}`);
        }
        this.tokenEmbeddingTable = tf.layers.embedding({ inputDim: vocabSize, outputDim: N_EMBD });
        this.positionEmbeddingTable = tf.layers.embedding({ inputDim: BLOCK_SIZE, outputDim: N_EMBD });
        this.blocks = Array.from({ length: N_LAYER }, () => new Block(N_EMBD, N_HEAD));
        this.lnF = tf.layers.layerNormalization();
        this.lmHead = tf.layers.dense({ units: vocabSize });
    }

    call(idx, training = false) {
        const T = idx.shape[1];
        const tokEmb = this.tokenEmbeddingTable.apply(idx);
        const posIndices = tf.range(0, T, 1, 'int32');
        const posEmb = this.positionEmbeddingTable.apply(posIndices).reshape([1, T, N_EMBD]);
        let x = tf.add(tokEmb, posEmb);
        this.blocks.forEach(block => {
            x = block.call(x, training);
        });
        x = this.lnF.apply(x);
        return this.lmHead.apply(x);
    }

    generate(idx, maxNewTokens) {
        for (let i = 0; i < maxNewTokens; i++) {
            const idxCond = idx.slice([0, Math.max(0, idx.shape[1] - BLOCK_SIZE)], [-1, -1]);
            let logits = this.call(idxCond, false);
            logits = logits.slice([0, logits.shape[1] - 1, 0], [-1, 1, -1]).reshape([-1, this.vocabSize]);
            const nextToken = tf.multinomial(logits, 1, undefined, 'int32');
            idx = tf.concat([idx, nextToken.reshape([1, 1])], 1);
        }
        return idx;
    }
}

export { GPTLanguageModel };
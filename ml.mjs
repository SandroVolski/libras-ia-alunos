// Rede neural pequena (MLP) para reconhecer formatos de mão, treinada no próprio navegador.
// Entrada: 21 pontos da mão (x, y, z) = 63 números. Saída: probabilidade de cada letra.

export const N_PTS = 21;
export const N_FEAT = N_PTS * 3;

// Gerador de números aleatórios com semente (resultados repetíveis)
export function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Transforma os 21 pontos da mão em 63 números que não dependem de onde a mão está
// nem do tamanho dela: o pulso vira a origem e a escala é o comprimento pulso -> base do dedo médio.
// aspect = largura/altura do vídeo; flip = true para mão esquerda (espelha, para as duas mãos serem iguais).
export function featuresFromLandmarks(lm, aspect, flip) {
  const w = lm[0];
  const ref = Math.hypot((lm[9].x - w.x) * aspect, lm[9].y - w.y) || 1e-6;
  const f = new Float32Array(N_FEAT);
  for (let i = 0; i < N_PTS; i++) {
    const x = (lm[i].x - w.x) * aspect / ref;
    f[i * 3] = flip ? -x : x;
    f[i * 3 + 1] = (lm[i].y - w.y) / ref;
    f[i * 3 + 2] = (lm[i].z - w.z) * aspect / ref;
  }
  return f;
}

// Cria variações pequenas de um exemplo (girar um pouco, mudar o tamanho, tremer)
// para a rede aprender a tolerar o jeito diferente de cada pessoa segurar a mão.
export function augment(f, rnd) {
  const a = (rnd() - 0.5) * 0.35;
  const s = 0.92 + rnd() * 0.16;
  const c = Math.cos(a), si = Math.sin(a);
  const g = new Float32Array(f.length);
  for (let i = 0; i < N_PTS; i++) {
    const x = f[i * 3], y = f[i * 3 + 1], z = f[i * 3 + 2];
    g[i * 3] = (c * x - si * y) * s + (rnd() - 0.5) * 0.03;
    g[i * 3 + 1] = (si * x + c * y) * s + (rnd() - 0.5) * 0.03;
    g[i * 3 + 2] = z * s + (rnd() - 0.5) * 0.03;
  }
  return g;
}

// Medidas extras calculadas a partir dos 63 números: distâncias entre a ponta do polegar e os outros dedos,
// e de cada ponta de dedo até o pulso/base. Ajudam a separar sinais parecidos (A x S, M x N x T, E x O...).
const PARES = [];
for (const j of [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20]) PARES.push([4, j]);
for (const j of [8, 12, 16, 20]) PARES.push([j, 0]);
for (const [a, b] of [[8, 5], [12, 9], [16, 13], [20, 17], [3, 6], [3, 10], [8, 12], [12, 16], [16, 20]]) PARES.push([a, b]);
export const N_EXTRA = PARES.length;
export function expand(f) {
  const g = new Float32Array(N_FEAT + N_EXTRA);
  g.set(f);
  for (let n = 0; n < PARES.length; n++) {
    const [a, b] = PARES[n];
    g[N_FEAT + n] = 2 * Math.hypot(f[a * 3] - f[b * 3], f[a * 3 + 1] - f[b * 3 + 1], (f[a * 3 + 2] - f[b * 3 + 2]) * 0.5);
  }
  return g;
}

export class MLP {
  constructor(nIn, nHid, nOut, seed = 7) {
    this.nIn = nIn; this.nHid = nHid; this.nOut = nOut;
    const rnd = mulberry32(seed);
    const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
    this.W1 = new Float32Array(nHid * nIn).map(() => gauss() * Math.sqrt(2 / nIn));
    this.b1 = new Float32Array(nHid);
    this.W2 = new Float32Array(nOut * nHid).map(() => gauss() * Math.sqrt(2 / nHid));
    this.b2 = new Float32Array(nOut);
    this.params = [this.W1, this.b1, this.W2, this.b2];
    this.m = this.params.map(p => new Float32Array(p.length));
    this.v = this.params.map(p => new Float32Array(p.length));
    this.t = 0;
  }

  // Palpite: devolve a probabilidade de cada classe
  predict(x) {
    if (this.expand) x = this.expand(x);
    const { nIn, nHid, nOut, W1, b1, W2, b2 } = this;
    const h = new Float32Array(nHid);
    for (let j = 0; j < nHid; j++) {
      let s = b1[j];
      for (let i = 0; i < nIn; i++) s += W1[j * nIn + i] * x[i];
      h[j] = s > 0 ? s : 0;
    }
    const z = new Float32Array(nOut);
    let zmax = -Infinity;
    for (let k = 0; k < nOut; k++) {
      let s = b2[k];
      for (let j = 0; j < nHid; j++) s += W2[k * nHid + j] * h[j];
      z[k] = s;
      if (s > zmax) zmax = s;
    }
    let sum = 0;
    for (let k = 0; k < nOut; k++) { z[k] = Math.exp(z[k] - zmax); sum += z[k]; }
    for (let k = 0; k < nOut; k++) z[k] /= sum;
    return z;
  }

  // Uma época: passa por todos os exemplos (com variações), mede o erro e ajusta os pesos.
  // Devolve o erro médio da época.
  trainEpoch(X, y, rnd, lr = 0.01, batch = 32, useAug = true) {
    const { nIn, nHid, nOut, W1, b1, W2, b2 } = this;
    const idx = Array.from({ length: X.length }, (_, i) => i);
    for (let i = idx.length - 1; i > 0; i--) {
      const j = Math.floor(rnd() * (i + 1));
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    let lossSum = 0;
    for (let start = 0; start < idx.length; start += batch) {
      const end = Math.min(start + batch, idx.length);
      const gW1 = new Float32Array(W1.length), gb1 = new Float32Array(nHid);
      const gW2 = new Float32Array(W2.length), gb2 = new Float32Array(nOut);
      for (let n = start; n < end; n++) {
        const k0 = idx[n];
        let x = useAug ? augment(X[k0], rnd) : X[k0];
        if (this.expand) x = this.expand(x);
        const h = new Float32Array(nHid);
        for (let j = 0; j < nHid; j++) {
          let s = b1[j];
          for (let i = 0; i < nIn; i++) s += W1[j * nIn + i] * x[i];
          h[j] = s > 0 ? s : 0;
        }
        const p = new Float32Array(nOut);
        let zmax = -Infinity;
        for (let k = 0; k < nOut; k++) {
          let s = b2[k];
          for (let j = 0; j < nHid; j++) s += W2[k * nHid + j] * h[j];
          p[k] = s;
          if (s > zmax) zmax = s;
        }
        let sum = 0;
        for (let k = 0; k < nOut; k++) { p[k] = Math.exp(p[k] - zmax); sum += p[k]; }
        for (let k = 0; k < nOut; k++) p[k] /= sum;
        lossSum += -Math.log(Math.max(p[y[k0]], 1e-9));
        const dh = new Float32Array(nHid);
        for (let k = 0; k < nOut; k++) {
          const g = p[k] - (k === y[k0] ? 1 : 0);   // quanto errou
          gb2[k] += g;
          for (let j = 0; j < nHid; j++) {
            gW2[k * nHid + j] += g * h[j];
            dh[j] += g * W2[k * nHid + j];
          }
        }
        for (let j = 0; j < nHid; j++) {
          if (h[j] <= 0) continue;
          gb1[j] += dh[j];
          for (let i = 0; i < nIn; i++) gW1[j * nIn + i] += dh[j] * x[i];
        }
      }
      this._adam([gW1, gb1, gW2, gb2], end - start, lr);
    }
    return lossSum / X.length;
  }

  // rede para os formatos de mão estáticos: 63 números + medidas extras
  static paraMao(nHid, nOut, seed) {
    const m = new MLP(N_FEAT + N_EXTRA, nHid, nOut, seed); m.expand = expand; return m;
  }

  _adam(grads, n, lr) {
    this.t++;
    const b1c = 1 - Math.pow(0.9, this.t), b2c = 1 - Math.pow(0.999, this.t);
    for (let q = 0; q < this.params.length; q++) {
      const p = this.params[q], g = grads[q], m = this.m[q], v = this.v[q];
      for (let i = 0; i < p.length; i++) {
        const gi = g[i] / n;
        m[i] = 0.9 * m[i] + 0.1 * gi;
        v[i] = 0.999 * v[i] + 0.001 * gi * gi;
        p[i] -= lr * (m[i] / b1c) / (Math.sqrt(v[i] / b2c) + 1e-8);
      }
    }
  }
}

// Reconhecimento de sinais com MOVIMENTO.
// Em vez de olhar uma foto parada da mão, olha uma janela de 1,2 s: o formato da mão
// em 8 instantes e o caminho que o pulso percorreu (para onde a mão foi).
import { featuresFromLandmarks } from './ml.mjs';

export const STEPS = 8;
export const DIM = 44;               // 21 pontos (x, y) = 42, mais o deslocamento do pulso (dx, dy)
export const FEAT = STEPS * DIM;     // 352 números por janela
export const WIN_MS = 1200;

// Resume um quadro da câmera: formato da mão (sem o tamanho/posição) e a posição do pulso.
export function frameInfo(lm, aspect, flip) {
  const f = featuresFromLandmarks(lm, aspect, flip);
  const shape = new Float32Array(42);
  for (let i = 0; i < 21; i++) { shape[i * 2] = f[i * 3]; shape[i * 2 + 1] = f[i * 3 + 1]; }
  const ref = Math.hypot((lm[9].x - lm[0].x) * aspect, lm[9].y - lm[0].y) || 1e-6;
  return { shape, wx: (flip ? -1 : 1) * lm[0].x * aspect, wy: lm[0].y, ref };
}

// Monta os 352 números de uma janela que termina em tEnd. Devolve null se faltarem quadros.
export function windowFeatures(frames, tEnd, win = WIN_MS) {
  const t0 = tEnd - win;
  const fr = frames.filter(f => f.t >= t0 && f.t <= tEnd);
  if (fr.length < 8) return null;
  if (fr[0].t - t0 > 300 || tEnd - fr[fr.length - 1].t > 300) return null;
  const refs = fr.map(f => f.ref).sort((a, b) => a - b), ref = refs[refs.length >> 1];
  const w0 = fr[0], out = new Float32Array(FEAT);
  for (let k = 0; k < STEPS; k++) {
    const tt = t0 + win * (k + 0.5) / STEPS;
    let best = fr[0], bd = Infinity;
    for (const f of fr) { const d = Math.abs(f.t - tt); if (d < bd) { bd = d; best = f; } }
    out.set(best.shape, k * DIM);
    out[k * DIM + 42] = (best.wx - w0.wx) / ref;
    out[k * DIM + 43] = (best.wy - w0.wy) / ref;
  }
  return out;
}

// Quanto o pulso andou nos últimos win ms, em "tamanhos de mão".
export function motion(frames, tEnd, win = 500) {
  const fr = frames.filter(f => f.t >= tEnd - win && f.t <= tEnd);
  if (fr.length < 3) return 0;
  let s = 0;
  for (let i = 1; i < fr.length; i++) s += Math.hypot(fr[i].wx - fr[i - 1].wx, fr[i].wy - fr[i - 1].wy) / fr[i].ref;
  return s;
}

// Quanto de movimento existe DENTRO de uma janela de 352 números: mudança do formato da mão + caminho do pulso.
export function windowMove(f) {
  let s = 0;
  for (let k = 1; k < STEPS; k++) {
    let sh = 0;
    for (let i = 0; i < 21; i++) sh += Math.hypot(f[k * DIM + i * 2] - f[(k - 1) * DIM + i * 2], f[k * DIM + i * 2 + 1] - f[(k - 1) * DIM + i * 2 + 1]);
    s += sh / 21 + Math.hypot(f[k * DIM + 42] - f[(k - 1) * DIM + 42], f[k * DIM + 43] - f[(k - 1) * DIM + 43]);
  }
  return s;
}

// Janela de "mão se ajeitando": o formato vai de um sinal qualquer até o sinal final (como quando você
// move a mão para fazer o I). Isso NÃO é um movimento de letra.
export function transitionWindow(a42, b42, rnd) {
  const out = new Float32Array(FEAT), c = 0.25 + rnd() * 0.4, dx = (rnd() - 0.5) * 1.2, dy = (rnd() - 0.5) * 1.2;
  for (let k = 0; k < STEPS; k++) {
    const u = k / (STEPS - 1), s = 1 / (1 + Math.exp(-(u - c) * 12));
    for (let i = 0; i < 42; i++) out[k * DIM + i] = a42[i] * (1 - s) + b42[i] * s + (rnd() - 0.5) * 0.03;
    out[k * DIM + 42] = dx * u * (1 - s * 0.3);
    out[k * DIM + 43] = dy * u * (1 - s * 0.3);
  }
  return out;
}

// Janelas de "mão parada", para a rede aprender o que NÃO é um movimento.
export function idleWindow(shape42, rnd) {
  const out = new Float32Array(FEAT);
  for (let k = 0; k < STEPS; k++) {
    for (let i = 0; i < 42; i++) out[k * DIM + i] = shape42[i] + (rnd() - 0.5) * 0.04;
    out[k * DIM + 42] = (rnd() - 0.5) * 0.05;
    out[k * DIM + 43] = (rnd() - 0.5) * 0.05;
  }
  return out;
}

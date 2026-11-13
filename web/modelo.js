// Inferencia del modelo final en JavaScript, equivalente al Pipeline de sklearn
// (src/comun.py: normalizar; TfidfVectorizer char_wb; LogisticRegression multinomial).
// tests/check_web.mjs verifica que las probabilidades coinciden con las de Python.

// Clases de caracteres de Python 3 (re con str): \w = alfanumérico Unicode o "_",
// \s = str.isspace(), \d = dígito decimal Unicode.
const W = "\\p{L}\\p{N}_";
const S = "\\t\\n\\v\\f\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const NO_W_ANTES = `(?<![${W}])`;
const NO_W_DESPUES = `(?![${W}])`;

const RE_URL = new RegExp(`https?://[^${S}]+`, "gu");
const RE_TOK = /<(correo|numero|nombre)>/gu;
const RE_ENTIDAD = new RegExp(
  `${NO_W_ANTES}(?:david ?p?lata|david ?plta|davi(?!d${NO_W_DESPUES})[${W}]*|nequi[${W}]*|nu ?bank|nuu?)${NO_W_DESPUES}`,
  "gu");
const RE_DIG = /\p{Nd}+/gu;
const RE_REP = /([^\n])\1{2,}/gu;
const RE_NO_PALABRA = new RegExp(`[^${W}ñ${S}]`, "gu");
const RE_ESP = new RegExp(`[${S}]+`, "gu");
const RE_ESP2 = new RegExp(`[${S}][${S}]+`, "gu");
const RE_BORDES = new RegExp(`^[${S}]+|[${S}]+$`, "gu");
const RE_MARCA = /\p{Mn}/u;

export function normalizar(t) {
  t = t.toLowerCase().normalize("NFKD");
  let s = "";
  for (const c of t) if (!RE_MARCA.test(c) || c === "̃") s += c;
  t = s.normalize("NFC");
  t = t.replace(RE_URL, " ");
  t = t.replace(RE_TOK, " $1tok ");
  t = t.replace(RE_ENTIDAD, " entidadtok ");
  t = t.replace(RE_DIG, " num ");
  t = t.replace(RE_REP, "$1$1");
  t = t.replace(RE_NO_PALABRA, " ");
  return t.replace(RE_ESP, " ").replace(RE_BORDES, "");
}

// sklearn _char_wb_ngrams (sobre puntos de código, como Python).
function ngramas(texto, nmin, nmax, f) {
  texto = texto.toLowerCase().replace(RE_ESP2, " ");
  for (const pal of texto.split(RE_ESP)) {
    if (!pal) continue;
    const w = Array.from(" " + pal + " ");
    for (let n = nmin; n <= nmax; n++) {
      let off = 0;
      f(w.slice(off, off + n).join(""));
      while (off + n < w.length) {
        off++;
        f(w.slice(off, off + n).join(""));
      }
      if (off === 0) break;
    }
  }
}

export class Modelo {
  constructor(json) {
    this.clases = json.clases;
    this.meta = json.meta;
    this.nmin = json.ngram_range[0];
    this.nmax = json.ngram_range[1];
    this.idx = new Map(json.vocab.map((g, i) => [g, i]));
    this.idf = json.idf;
    this.coef = json.coef;
    this.intercept = json.intercept;
    this.cache = new Map();
  }

  // Probabilidades de una reseña cruda (aplica la normalización, como el Pipeline).
  proba(crudo) {
    const norm = normalizar(String(crudo));
    if (this.cache.has(norm)) return this.cache.get(norm);
    const tf = new Map();
    ngramas(norm, this.nmin, this.nmax, (g) => {
      const i = this.idx.get(g);
      if (i !== undefined) tf.set(i, (tf.get(i) || 0) + 1);
    });
    const ids = [...tf.keys()].sort((a, b) => a - b);
    const val = ids.map((i) => (1 + Math.log(tf.get(i))) * this.idf[i]);
    const nrm = Math.sqrt(val.reduce((s, v) => s + v * v, 0));
    const z = this.intercept.map((b, k) => {
      let s = 0;
      ids.forEach((i, j) => { s += this.coef[k][i] * (nrm > 0 ? val[j] / nrm : 0); });
      return s + b;
    });
    const m = Math.max(...z);
    const e = z.map((v) => Math.exp(v - m));
    const tot = e.reduce((a, b) => a + b, 0);
    const p = e.map((v) => v / tot);
    if (this.cache.size > 20000) this.cache.clear();
    this.cache.set(norm, p);
    return p;
  }

  predecir(crudo) {
    const p = this.proba(crudo);
    const k = p.indexOf(Math.max(...p));
    return { clase: this.clases[k], k, p };
  }

  // Aportes de Shapley por palabra: una palabra "ausente" se borra del texto, como
  // la máscara de texto de SHAP (sus puntos suspensivos desaparecen al normalizar).
  // Exacto hasta 12 palabras; con más, se estima con permutaciones antitéticas.
  // Los aportes más la probabilidad base (texto vacío) suman la predicción.
  explicar(crudo, presupuesto = 6000) {
    const partes = String(crudo).split(new RegExp(`([${W}]+)`, "u"));
    const pos = [];
    partes.forEach((p, i) => { if (i % 2 === 1) pos.push(i); });
    const n = pos.length;
    const f = (presente) => {
      const q = partes.slice();
      pos.forEach((i, j) => { if (!presente[j]) q[i] = " "; });
      return this.proba(q.join(""));
    };
    const K = this.clases.length;
    const phi = Array.from({ length: n }, () => new Array(K).fill(0));
    const base = f(new Array(n).fill(false));
    let metodo;
    if (n === 0) {
      metodo = "exacto";
    } else if (n <= 12) {
      metodo = "exacto";
      const valor = new Array(1 << n);
      for (let m = 0; m < 1 << n; m++) valor[m] = f(pos.map((_, j) => (m >> j) & 1));
      const fact = [1];
      for (let i = 1; i <= n; i++) fact[i] = fact[i - 1] * i;
      const pc = (m) => { let c = 0; while (m) { c += m & 1; m >>= 1; } return c; };
      for (let m = 0; m < 1 << n; m++) {
        const s = pc(m);
        for (let j = 0; j < n; j++) {
          if ((m >> j) & 1) continue;
          const w = (fact[s] * fact[n - s - 1]) / fact[n];
          const a = valor[m | (1 << j)], b = valor[m];
          for (let k = 0; k < K; k++) phi[j][k] += w * (a[k] - b[k]);
        }
      }
    } else {
      metodo = "permutaciones";
      const pares = Math.max(4, Math.floor(presupuesto / (2 * n)));
      let semilla = 42;
      const azar = () => ((semilla = (semilla * 1103515245 + 12345) % 2147483648) / 2147483648);
      const recorrer = (orden) => {
        const presente = new Array(n).fill(false);
        let prev = base;
        for (const j of orden) {
          presente[j] = true;
          const act = f(presente);
          for (let k = 0; k < K; k++) phi[j][k] += act[k] - prev[k];
          prev = act;
        }
      };
      for (let r = 0; r < pares; r++) {
        const orden = [...Array(n).keys()];
        for (let i = n - 1; i > 0; i--) {
          const j = Math.floor(azar() * (i + 1));
          [orden[i], orden[j]] = [orden[j], orden[i]];
        }
        recorrer(orden);
        recorrer(orden.slice().reverse());
      }
      phi.forEach((v) => v.forEach((_, k) => { v[k] /= 2 * pares; }));
    }
    return {
      palabras: pos.map((i, j) => ({ palabra: partes[i], aporte: phi[j] })),
      base, metodo,
    };
  }
}

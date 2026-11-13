import { Modelo } from "./modelo.js";

const EJEMPLOS = [
  "Es buena pero se cae mucho",
  "La app es muy práctica, aunque últimamente se demora mucho en abrir",
  "Pésimo servicio, llevo tres días sin poder retirar mi plata",
  "Me encanta, las transferencias llegan al instante y la tarjeta es gratis",
  // Caso límite conocido: contraste implícito, sin "pero"; el modelo lo lee como negativa.
  "Funcionaba de maravilla, desde la actualización no abre",
];

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const coma = (x, d = 2) => x.toFixed(d).replace(".", ",");
const pct = (x) => `${Math.round(x * 100)} %`;

let modelo;
let ultimo = null;

// ------------------------------------------------------------ pestañas
const tabs = [$("tab-una"), $("tab-csv")];
tabs.forEach((t, i) => {
  t.addEventListener("click", () => activar(i));
  t.addEventListener("keydown", (e) => {
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      const j = (i + (e.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
      activar(j); tabs[j].focus();
    }
  });
});
function activar(i) {
  tabs.forEach((t, j) => {
    t.setAttribute("aria-selected", String(i === j));
    t.tabIndex = i === j ? 0 : -1;
    $(t.getAttribute("aria-controls")).hidden = i !== j;
  });
}

// ------------------------------------------------------------ una reseña
EJEMPLOS.forEach((t) => {
  const b = document.createElement("button");
  b.type = "button";
  b.textContent = t;
  b.addEventListener("click", () => { $("texto").value = t; clasificar(); });
  $("ejemplos").appendChild(b);
});
$("clasificar").addEventListener("click", clasificar);
$("texto").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); clasificar(); }
});

function clasificar() {
  if (!modelo) return;
  const texto = $("texto").value;
  const out = $("resultado");
  out.hidden = false;
  if (!texto.trim()) {
    out.innerHTML = `<p class="error">Escribe una reseña.</p>`;
    ultimo = null;
    return;
  }
  const { clase, k, p } = modelo.predecir(texto);
  const exp = modelo.explicar(texto);
  ultimo = { texto, clase, k, p, exp };
  const corta = texto.trim().split(/\s+/).length < 3;
  out.innerHTML = `
    <h2>Predicción: <span class="pildora p-${clase}">${clase}</span></h2>
    ${corta ? `<p class="aviso">El modelo se entrenó con reseñas de tres palabras o más; con textos tan cortos la predicción es poco confiable.</p>` : ""}
    <div class="barras" role="list" aria-label="Probabilidad de cada clase">
      ${modelo.clases.map((c, j) => `
        <div class="barra" role="listitem">
          <span>${c}</span>
          <div class="pista"><div class="relleno f-${c}" style="width:${(p[j] * 100).toFixed(1)}%"></div></div>
          <span class="num">${pct(p[j])}</span>
        </div>`).join("")}
    </div>
    <div class="explica">
      <h3>Por qué</h3>
      <div class="selector" role="group" aria-label="Clase explicada">
        <span>Aporte a la probabilidad de</span>
        ${modelo.clases.map((c, j) => `<button type="button" data-k="${j}" aria-pressed="${j === k}">${c}</button>`).join("")}
      </div>
      <div class="palabras" id="palabras"></div>
      <div class="leyenda"><span>baja la probabilidad</span><span class="degradado" aria-hidden="true"></span><span>la sube</span></div>
      <details>
        <summary>Aportes de Shapley por palabra y clase</summary>
        <div class="tabla-scroll" id="tabla-aportes"></div>
      </details>
      <p class="metodo">Cada palabra se quita del texto y se mide cuánto cambia la probabilidad (valores de Shapley, ${exp.metodo === "exacto" ? "cálculo exacto" : "estimados con permutaciones"}). Probabilidad base con el texto vacío más la suma de aportes = probabilidad predicha.</p>
    </div>`;
  out.querySelectorAll(".selector button").forEach((b) =>
    b.addEventListener("click", () => {
      out.querySelectorAll(".selector button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      pintar(Number(b.dataset.k));
    }));
  pintar(k);
  $("tabla-aportes").innerHTML = `<table>
    <thead><tr><th>Palabra</th>${modelo.clases.map((c) => `<th class="n">${c}</th>`).join("")}</tr></thead>
    <tbody>
      <tr><td><i>base (texto vacío)</i></td>${exp.base.map((v) => `<td class="n">${coma(v, 3)}</td>`).join("")}</tr>
      ${exp.palabras.map((w) => `<tr><td>${esc(w.palabra)}</td>${w.aporte.map((v) => `<td class="n">${v >= 0 ? "+" : "−"}${coma(Math.abs(v), 3)}</td>`).join("")}</tr>`).join("")}
    </tbody></table>`;
}

function pintar(k) {
  const { texto, exp } = ultimo;
  const tope = Math.max(0.05, ...exp.palabras.map((w) => Math.abs(w.aporte[k])));
  // Reconstruye el texto original, coloreando solo las palabras.
  const partes = texto.split(/([\p{L}\p{N}_]+)/u);
  let j = 0;
  $("palabras").innerHTML = partes.map((p, i) => {
    if (i % 2 === 0) return esc(p);
    const v = exp.palabras[j++].aporte[k];
    const a = Math.min(Math.abs(v) / tope, 1) * 0.75;
    const col = v >= 0 ? "var(--sube)" : "var(--baja)";
    const signo = v >= 0 ? "+" : "−";
    return `<span class="w" style="background:rgba(${col},${a.toFixed(2)})" title="${signo}${coma(Math.abs(v), 3)}">${esc(p)}</span>`;
  }).join("");
}

// ------------------------------------------------------------ CSV
function leerCSV(txt) {
  txt = txt.replace(/^﻿/, "");
  const primera = txt.split(/\r?\n/, 1)[0];
  const sep = !primera.includes(",") && primera.includes(";") ? ";" : ",";
  const filas = [];
  let fila = [], campo = "", comillas = false;
  for (let i = 0; i < txt.length; i++) {
    const c = txt[i];
    if (comillas) {
      if (c === '"') {
        if (txt[i + 1] === '"') { campo += '"'; i++; } else comillas = false;
      } else campo += c;
    } else if (c === '"') comillas = true;
    else if (c === sep) { fila.push(campo); campo = ""; }
    else if (c === "\n" || c === "\r") {
      if (c === "\r" && txt[i + 1] === "\n") i++;
      fila.push(campo); filas.push(fila); fila = []; campo = "";
    } else campo += c;
  }
  if (campo !== "" || fila.length) { fila.push(campo); filas.push(fila); }
  return filas.filter((f) => !(f.length === 1 && f[0] === ""));
}

const celdaCSV = (v) => (/[",\n\r]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v);

async function procesar(archivo) {
  const err = $("csv-error"), out = $("csv-resultado");
  err.hidden = true; out.hidden = true;
  if (!modelo) return;
  const filas = leerCSV(await archivo.text());
  if (!filas.length) { err.textContent = "El archivo está vacío."; err.hidden = false; return; }
  const cab = filas[0].map((c) => c.trim());
  const col = cab.indexOf("texto");
  if (col < 0) { err.textContent = "El archivo no tiene una columna llamada texto."; err.hidden = false; return; }
  const datos = filas.slice(1);
  const res = datos.map((f) => modelo.predecir(f[col] ?? ""));
  const cuenta = Object.fromEntries(modelo.clases.map((c) => [c, 0]));
  res.forEach((r) => cuenta[r.clase]++);
  const nuevaCab = [...cab, "prediccion", ...modelo.clases.map((c) => `p_${c}`)];
  const lineas = [nuevaCab, ...datos.map((f, i) => [...cab.map((_, j) => f[j] ?? ""), res[i].clase, ...res[i].p.map((v) => v.toFixed(3))])];
  const csv = "﻿" + lineas.map((l) => l.map(celdaCSV).join(",")).join("\r\n");
  const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
  const n = datos.length, MAX = 100;
  out.innerHTML = `
    <div class="fila-csv">
      <h3>${n.toLocaleString("es-CO")} reseñas clasificadas</h3>
      <a class="primario" href="${url}" download="predicciones.csv" style="text-decoration:none">Descargar resultados</a>
    </div>
    <div class="resumen">${modelo.clases.map((c) => `<div class="r-${c}"><b>${cuenta[c].toLocaleString("es-CO")}</b><span>${c} · ${n ? pct(cuenta[c] / n) : "0 %"}</span></div>`).join("")}</div>
    <div class="tabla-scroll"><table>
      <thead><tr><th>Texto</th><th>Predicción</th>${modelo.clases.map((c) => `<th class="n">p ${c}</th>`).join("")}</tr></thead>
      <tbody>${datos.slice(0, MAX).map((f, i) => `<tr><td class="texto">${esc(f[col] ?? "")}</td><td><span class="pildora p-${res[i].clase}">${res[i].clase}</span></td>${res[i].p.map((v) => `<td class="n">${coma(v)}</td>`).join("")}</tr>`).join("")}</tbody>
    </table></div>
    ${n > MAX ? `<p class="metodo">Se muestran las primeras ${MAX} filas; el archivo descargado las tiene todas.</p>` : ""}`;
  out.hidden = false;
}

$("archivo").addEventListener("change", (e) => { if (e.target.files[0]) procesar(e.target.files[0]); });
const zona = $("zona");
["dragenter", "dragover"].forEach((ev) => zona.addEventListener(ev, (e) => { e.preventDefault(); zona.classList.add("encima"); }));
["dragleave", "drop"].forEach((ev) => zona.addEventListener(ev, (e) => { e.preventDefault(); zona.classList.remove("encima"); }));
zona.addEventListener("drop", (e) => { if (e.dataTransfer.files[0]) procesar(e.dataTransfer.files[0]); });

// ------------------------------------------------------------ carga
(async () => {
  const boton = $("clasificar");
  try {
    const r = await fetch("model.json");
    modelo = new Modelo(await r.json());
  } catch (e) {
    boton.textContent = "No se pudo cargar el modelo";
    return;
  }
  const pt = modelo.meta.prueba_temporal;
  $("metricas").innerHTML = [
    ["F1 macro", coma(pt.f1_macro)],
    ["F1 de conflicto", coma(pt.f1_conflicto)],
    ["Exactitud", coma(pt.exactitud)],
    ["Conflictos hallados", "20 de 26"],
  ].map(([a, b]) => `<div><dt>${a}</dt><dd>${b}</dd></div>`).join("");
  boton.textContent = "Clasificar";
  boton.disabled = false;
})();

"""Depuración, anonimización y etiqueta débil.

Pasos (en este orden, todos independientes de la variable objetivo):
 1. Descartar reseñas vacías.
 2. Anonimizar correos, teléfonos, cédulas, números de cuenta (>=7 dígitos)
    y nombres propios introducidos con "mi nombre es", "me llamo" o "soy".
 3. Descartar reseñas con menos de MIN_PALABRAS palabras (no aportan texto
    suficiente para distinguir conflicto y casi siempre son duplicados).
 4. Eliminar duplicados exactos del texto normalizado, para que un mismo
    texto no quede a la vez en entrenamiento y en prueba.
 5. Filtrar reseñas en inglés o portugués (regla de palabras funcionales).
 6. Descartar reseñas de Nu con marcadores de México (INE, CURP, Sofipo...).
 7. Etiqueta débil: 1-2 estrellas -> negativa, 4-5 -> positiva,
    3 -> pendiente de anotación manual.

Salida: data/processed/reviews_clean.csv y un resumen del embudo.
"""
from pathlib import Path
import re
import unicodedata

import pandas as pd
MIN_PALABRAS = 3

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "reviews_raw.csv"
OUT = ROOT / "data" / "processed" / "reviews_clean.csv"
EMBUDO = ROOT / "reports" / "embudo_depuracion.csv"

RE_EMAIL = re.compile(r"\S+@\S+\.\S+")
RE_NUM = re.compile(r"(?:\+?\d[\d\s.-]{5,}\d)")
RE_NOMBRE = re.compile(r"(mi nombre es|me llamo|soy)\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){1,3})")
# La app de Nu es global: con country="co" se cuelan reseñas de México.
RE_MEXICO = re.compile(
    r"\b(?:m[eé]xico|mexican[oa]s?|ine|curp|rfc|sofipos?|klar|elektra|coppel|zacatecas|"
    r"cdmx|spei|oxxo|telcel|msi|hsbc|bur[oó] de cr[eé]dito|la neta|chido|nip)\b")


def anonimizar(t: str) -> str:
    t = RE_EMAIL.sub("<correo>", t)
    t = RE_NOMBRE.sub(lambda m: f"{m.group(1)} <nombre>", t)
    return RE_NUM.sub(lambda m: "<numero>" if sum(c.isdigit() for c in m.group()) >= 7
                      else m.group(), t)


def normalizar(t: str) -> str:
    """Clave para detectar duplicados: minúsculas, sin tildes ni puntuación."""
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-zñ0-9 ]+", " ", t).split())


# langdetect descartaba mucho español válido en textos cortos ("mala no abre"),
# así que uso una regla de palabras funcionales: solo descarto la reseña si
# tiene más marcadores de inglés o portugués que de español.
ES = set("de la el que y en no me es lo un una por con para muy se mi pero ya mas más todo app".split())
EN = set("the and is it to you this for not my very good great but with of app".split()) - {"app"}
PT = set("não muito uma está você com isso aplicativo ótimo obrigado então".split())


def es_espanol(t: str) -> bool:
    toks = re.findall(r"[a-záéíóúñãõç]+", t.lower())
    n_es = sum(w in ES for w in toks)
    n_otro = max(sum(w in EN for w in toks), sum(w in PT for w in toks))
    return not (n_otro >= 2 and n_otro > n_es)


def depurar(df: pd.DataFrame):
    """Aplica los pasos 1 a 6 y devuelve (reseñas depuradas, embudo)."""
    embudo = [("extraídas", len(df))]

    df = df[df.texto.notna() & df.texto.str.strip().ne("")].copy()
    embudo.append(("con texto", len(df)))

    df["texto"] = df.texto.str.strip().map(anonimizar)
    df["n_palabras"] = df.texto.str.split().str.len()
    df = df[df.n_palabras >= MIN_PALABRAS]
    embudo.append((f">= {MIN_PALABRAS} palabras", len(df)))

    df["clave"] = df.texto.map(normalizar)
    df = df.drop_duplicates("clave").drop(columns="clave")
    embudo.append(("sin duplicados", len(df)))

    df = df[df.texto.map(es_espanol)]
    embudo.append(("en español", len(df)))

    mx = (df.entidad == "Nu") & df.texto.str.lower().str.contains(RE_MEXICO)
    df = df[~mx]
    embudo.append(("sin reseñas de México", len(df)))
    return df, embudo


def main():
    df, embudo = depurar(pd.read_csv(RAW))

    df["etiqueta_debil"] = df.estrellas.map(
        {1: "negativa", 2: "negativa", 3: "pendiente", 4: "positiva", 5: "positiva"})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    EMBUDO.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(embudo, columns=["paso", "n"]).to_csv(EMBUDO, index=False)
    for paso, n in embudo:
        print(f"{paso:>20}: {n}")
    print(pd.crosstab(df.entidad, df.estrellas, margins=True))


if __name__ == "__main__":
    main()

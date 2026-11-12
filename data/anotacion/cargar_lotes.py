"""Une los lotes de preetiquetas (formato: 'inicio ETIQUETAS') en un CSV."""
from pathlib import Path
import pandas as pd
D = Path(__file__).parent
MAP = {"P": "positiva", "N": "negativa", "C": "conflicto", "I": "irrelevante"}
filas = []
for f in sorted((D / "lotes").glob("l*.txt")):
    for linea in f.read_text().split("\n"):
        if not linea.strip():
            continue
        ini, et = linea.split()
        assert len(et) == 10 or f == max((D / "lotes").glob("l*.txt")), (f, linea)
        filas += [(int(ini) + k, MAP[c]) for k, c in enumerate(et)]
df = pd.DataFrame(filas, columns=["fila", "preetiqueta"])
assert df.fila.is_unique and (df.fila.diff().dropna() == 1).all(), "huecos o duplicados"
df.to_csv(D / "preetiquetas.csv", index=False)
print(len(df), df.preetiqueta.value_counts().to_dict())

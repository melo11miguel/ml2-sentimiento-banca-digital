# Guía de anotación: polaridad de reseñas de banca digital

Cada reseña recibe UNA etiqueta según lo que dice el texto. No miren la calificación
en estrellas (las hojas no la muestran a propósito).

| Etiqueta | Cuándo usarla | Ejemplos |
|---|---|---|
| **positiva** | Valoración favorable sin ninguna queja. | "muy buena app fácil de usar"; "excelente servicio, gracias" |
| **negativa** | Queja, falla, reclamo o pedido de ayuda por un problema, sin valoración positiva explícita. Incluye exigencias de mejora ("deberían bajar los intereses"). | "no me deja ingresar"; "pésima, se cae a cada rato"; "me pide actualizar y no abre, ayuda" |
| **conflicto** | En la misma reseña hay una valoración positiva explícita **y** una queja. | "es buena pero se cae mucho"; "excelente app, deberían mejorar la interfaz"; "funcionaba de maravilla, desde la actualización no abre"; "nunca había tenido problemas, pero hoy no entra" |
| **irrelevante** | No expresa valoración del servicio: preguntas informativas, instrucciones para otros usuarios, texto sin sentido o fuera de tema. Se descarta. | "¿cómo cambio mi número?"; "para los que no les abre, desactiven opciones de desarrollador" |

## Reglas para casos dudosos

1. **Pasado positivo + presente negativo = conflicto**, solo si el pasado es una
   valoración ("funcionaba bien", "era excelente", "nunca había tenido problemas").
   Un dato neutro ("antes ingresaba normal", "llevo dos años usándola") no cuenta: es **negativa**.
2. **Condicionales no son valoraciones positivas.** "Si no fallara sería genial" o
   "podría ser buena si..." son **negativa**.
3. **Pedidos de ayuda por un problema** ("no me deja entrar, ¿qué hago?") son **negativa**.
   Preguntas sin problema ("¿cómo pido la tarjeta de crédito?") son **irrelevante**.
4. **Positiva + sugerencia sin queja** ("me encanta, ojalá agreguen modo oscuro"):
   si la sugerencia señala una carencia que molesta, **conflicto**; si es un deseo
   sin reproche, **positiva**.
5. Si dudan entre dos etiquetas después de releer, elijan la que mejor describa
   la intención principal y marquen la columna `duda` con una x.

## Procedimiento

1. **Hoja ciega** (`hoja_ciega_<nombre>.xlsx`, 198 reseñas): cada uno la anota
   **por separado, sin hablarlo y sin ver la preetiqueta**. De aquí sale el kappa.
2. **Hoja de revisión** (`hoja_revision.xlsx`, 1 242 reseñas): trae una preetiqueta
   asistida por IA. Revisen cada fila y corrijan la columna `etiqueta` donde no
   estén de acuerdo. Pueden repartirse las filas.
3. Ejecuten `python src/07_kappa.py`: calcula el kappa de Cohen y lista los
   desacuerdos de la hoja ciega. Resuelvan cada desacuerdo por consenso en la
   columna `consenso` de `desacuerdos.xlsx` y vuelvan a ejecutar el script; ese
   segundo paso genera `etiquetas_finales.csv`.

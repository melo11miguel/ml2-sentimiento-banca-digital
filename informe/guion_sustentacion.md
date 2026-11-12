# Guion de sustentación: entrega final

Duración objetivo: 12 minutos de exposición y demostración, más preguntas. Cada parte indica quién
la presenta, pero los dos deben poder explicar cualquiera: el profesor puede preguntarle a
cualquiera de los dos.

## 1. El artículo y el problema (Miguel, 2 min)

- Alawaji y Aloraini (2025) clasifican reseñas de banca digital saudí en tres clases: positiva,
  negativa y **conflicto** (elogio y queja en la misma reseña). Su mejor modelo, un voto con
  ponderación de clases, llega a una exactitud de 0,90.
- Nuestra pregunta: ¿se puede hacer lo mismo con reseñas colombianas (Nequi, DaviPlata, Nu) y
  qué modelo equilibra el desempeño global con la detección del conflicto?
- La crítica central desde la Entrega 1: el artículo nunca reporta la clase conflicto por separado.
  Con un 4 % de conflicto, una exactitud del 90 % se logra ignorándolo.

## 2. Datos (Nicolás, 2 min)

- 30 000 reseñas de Google Play, que quedan en 19 747 tras depurar (anonimización, menos de tres
  palabras, duplicados, otros idiomas y reseñas de México que se colaban por Nu).
- Etiqueta débil por estrellas, verificada con una muestra anotada: 4 estrellas como positiva
  acierta solo el 24 %, así que la descartamos; el error de la etiqueta débil que sí usamos es 16,8 %.
- Entrega final: **prueba temporal**. 600 reseñas publicadas después del entrenamiento (1 de
  octubre a 11 de noviembre), 200 por entidad, con la distribución natural de clases. La partimos por
  fecha: octubre para validar una decisión, y del 30 de octubre en adelante (291 reseñas) como prueba
  que usamos una sola vez.

## 3. Flujo sin fuga y selección (Miguel, 2 min)

- Todo lo que aprende de los datos (vocabulario TF-IDF, IDF, chi cuadrado, SMOTE, modelo) va dentro
  del `Pipeline` y se ajusta solo con el pliegue de entrenamiento.
- CV anidada: 5 pliegues externos y 3 internos. Gana la regresión logística sobre n-gramas de
  caracteres (0,724); el stacking empata (0,718) pero tarda tres veces más y no se interpreta.
- Enmascaramos los nombres de las entidades para que "nu" no sea señal de elogio. No cuesta nada:
  0,889 con máscara frente a 0,892 sin ella.
- La validación temporal decidió entrenar con oro **y** etiqueta débil (0,840 frente a 0,790 solo
  oro), al revés que la CV sobre el oro. Razón: el oro está lleno de reseñas de 3 estrellas,
  ambiguas; en el uso real dominan las de 1 y 5 estrellas, donde la etiqueta débil acierta.

## 4. Resultados y contraste con el artículo (Nicolás, 2 min)

- Prueba temporal: F1 macro 0,889 (IC 95 %: 0,83 a 0,94), F1 de conflicto 0,78, recupera 20 de 26.
- El protocolo del artículo en los mismos datos: exactitud 0,894 (su 0,90) pero F1 de conflicto
  0,14, solo 2 de 26. Reproducimos su resultado **y** su punto ciego.
- Confirmamos: la ponderación de clases es indispensable. No se sostiene: quitar palabras vacías
  (borra "pero" y "no") y el voto por mayoría para la clase minoritaria.

## 5. Interpretación con SHAP (Miguel, 2 min)

- Figura 8: el conflicto depende casi por completo de "pero" (aporte medio 0,37), seguido de
  "solo", "único", "bien". El modelo aprendió la forma: elogio tibio más restricción.
- Figura 9, los errores: un "excelente" muy fuerte tapa la queja; el saludo "buenas" se lee como
  elogio; la negación ("no estoy feliz") no se capta; el "pero" intensificador engaña.
- Uso práctico: de cada cinco reseñas que marca como conflicto, cuatro lo son, y encuentra tres de
  cada cuatro. Sirve para priorizar, no para decidir solo.

## 6. Demostración (Nicolás, 1,5 min)

Antes de la sustentación, abrir la terminal en la carpeta del proyecto y lanzar:

```bash
.venv/bin/streamlit run app/app.py
```

1. Elegir el ejemplo "Es buena pero se cae mucho": sale conflicto con probabilidad alta; mostrar
   "pero" en verde.
2. Elegir "Pésimo servicio, llevo tres días sin poder retirar mi plata": negativa.
3. Elegir "Funcionaba de maravilla, desde la actualización no abre": sale **negativa**, aunque la guía
   lo trata como conflicto. Decirlo abiertamente: es el límite de una bolsa de n-gramas, no ve el
   contraste sin conector. Es el puente a la proyección con BETO.
4. Pestaña CSV: subir `reports/predicciones_prueba.csv` (tiene columna `texto`) y descargar.

## 7. Limitaciones y proyección (Miguel, 0,5 min)

- Las etiquetas son preetiquetas de IA sin revisión humana completa ni kappa: el desempeño medido es
  el acuerdo con un criterio consistente; con anotación humana probablemente bajaría.
- Prueba pequeña (26 conflictos) y de seis semanas.
- Deep Learning: ajustar BETO para negación y contraste implícito. MLOps: monitorear deriva de
  vocabulario tras cada actualización, anotar periódicamente con aprendizaje activo, versionar y
  reentrenar.

## Preguntas probables y respuesta corta

**¿Por qué la prueba da 0,89 y la CV 0,72?** Porque miden cosas distintas. La CV se puntúa sobre el
oro, que tiene todas las reseñas de 3 estrellas (las más ambiguas). La prueba tiene la mezcla real.
0,89 es el desempeño esperado en uso; 0,72, en los casos difíciles.

**¿No eligieron el modelo mirando la prueba?** No. La familia salió de la CV anidada, la variante de
entrenamiento de la validación temporal (octubre) y la prueba (desde el 30 de octubre) se usó una
sola vez. La regla está escrita en `src/14_evaluacion_temporal.py`.

**¿Por qué no el stacking, si dio 0,896?** La diferencia pareada con la regresión logística va de
−0,03 a 0,04: no se distingue. Es 18 veces más lento y no se interpreta directamente.

**¿Dónde podría haber fuga?** En el TF-IDF ajustado con todo el corpus (lo evitamos con el
Pipeline), en duplicados entre entrenamiento y prueba (los quitamos antes de partir y quitamos de la
prueba temporal los textos que ya estaban en el corpus) y en usar las estrellas como predictor (de
ahí sale la etiqueta débil; nunca entran al modelo).

**¿Por qué n-gramas de caracteres?** Son robustos a la ortografía informal ("exelente", "aveces",
"dsviplata") y a variantes de una misma palabra; en el experimento de representación ganaron a
unigramas y bigramas.

**¿Qué hace SHAP exactamente aquí?** Quita palabras completas de la reseña y mide cuánto cambia la
probabilidad de cada clase. Los aportes más la probabilidad base suman exactamente la predicción.

**¿Por qué ponderación de clases y no SMOTE?** Rinden parecido en la regresión logística, y SMOTE
fabrica vectores TF-IDF que no corresponden a ningún texto real.

**¿Cómo saben que la app predice igual que el modelo evaluado?** El modelo guardado incluye la
normalización dentro del Pipeline, y al guardarlo el script comprueba que sus predicciones sobre las
291 reseñas de prueba son idénticas a las evaluadas; si no, se detiene.

**¿Quién etiquetó?** Las etiquetas son preetiquetas asistidas por IA con una guía escrita, leyendo
solo el texto (sin estrellas ni predicciones). Lo declaramos y es nuestra limitación principal.

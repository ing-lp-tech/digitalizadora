# Benchmark: tiempo y memoria por foto procesada

Responde a los puntos 2.2 y 6.2 de `CORRECCIONES-HITO2.md` ("¿cuánto tarda y
cuánta memoria usa procesar una foto en el servidor?").

Medido sobre la misma foto (`IMG_20260909_184214.jpg` / `20260909_184126.jpg`,
14 marcadores, resultado `pass`), en dos entornos.

## Local (referencia sin restricciones de plan)

Script: `scripts/benchmark_motor.py`, 5 repeticiones sobre
`datos_ejemplo_fotos_aruco/20260909_184126.jpg`.

| Métrica | Valor |
|---|---|
| Tiempo por foto | ~0.3 s (min 0.28s, max 0.39s) |
| RSS del proceso | ~49 MB (creció ~13 MB sobre el baseline del intérprete) |
| Pico memoria Python (tracemalloc) | ~91 MB |

## Servidor real (Render, free tier, `WEB_CONCURRENCY=1`)

Medido con logging agregado a `web_editor/server.py` (tiempo con
`time.perf_counter()`, memoria con `resource.getrusage().ru_maxrss`),
`POST /api/process` sobre `IMG_20260909_184214.jpg`, dos fotos consecutivas:

| Métrica | Valor |
|---|---|
| Tiempo por foto | 8.09 s / 7.21 s |
| RSS pico del proceso | 416.8 MB / 454.7 MB |

`ru_maxrss` es el pico acumulado desde que arrancó el proceso, no memoria
aislada por request — el primer valor después de un restart es el más
representativo de lo que consume una foto sola.

## Lectura

- **CPU**: ~25x más lento que local. Esperable en el free tier de Render
  (CPU compartida, no dedicada).
- **Memoria — hallazgo no pedido explícitamente, pero relevante**: una sola
  foto ya usa ~450 MB. El plan free de Render da **512 MB de RAM** por
  instancia. Con `ThreadingHTTPServer`, cada foto que se procesa corre en un
  thread dentro del mismo proceso — si 2 o 3 personas procesan a la vez (el
  escenario de la sección 2.3 de `CORRECCIONES-HITO2.md`), es probable que la
  suma supere los 512 MB y Render mate el proceso por falta de memoria
  (OOM), no por el servidor bloqueado — eso último ya está resuelto
  (`ThreadingHTTPServer` confirmado).

## Recomendación antes de probar concurrencia real (punto 2.3)

No probar 3 personas procesando a la vez en el plan actual sin antes: subir
de plan (más RAM), o reducir el pico de memoria por foto (liberar imágenes
intermedias — overlays, máscaras — antes de responder, en vez de mantenerlas
todas en memoria del mismo proceso que sirve el resto de la app).

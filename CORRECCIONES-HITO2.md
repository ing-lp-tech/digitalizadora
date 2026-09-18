# Digitalizadora — correcciones del hito 2

Revisión del PR #1 (motor) y de lo publicado en `moldes.crevia.com.ar`.
**Son correcciones sobre lo entregado, no alcance nuevo.**

La app web, según la documentación del repo, corresponde al hito 3 y no formaba
parte de esta entrega. Está hecha antes de tiempo y es bienvenido; pero al estar
desplegada y en uso, lo que se observa acá es lo que falta para que quede
entregable.

**Contexto de uso:** fotos siempre con celular. Moldes de papel o cartón fino,
1 mm de espesor máximo. Hasta 3 personas procesando al mismo tiempo.

---

## 1. Bloqueantes

**1.1** El código de la app web no está en el repo: falta el frontend, el
servidor, `/api/process`, `/api/export` y los exportadores DXF y PDF. Hoy existe
solo en un deploy. → Subir todo en un PR.

**1.2** Que el PR incluya `Dockerfile` y `render.yaml`, para recrear el servicio
sin configuraciones hechas a mano en el panel.

**1.3** Runbook en el README: paso a paso para desplegar desde una cuenta vacía
(versión de Python, build, arranque, variables de entorno, disco, dominio,
Cloudflare). Criterio: que otra persona lo siga sin preguntar nada.

**1.4** Pasar Render y el dominio a una cuenta de la empresa, con acceso de
colaborador para el desarrollador. Dejar por escrito qué servicios existen y
quién los administra.

---

## 2. Concurrencia

El backend usa `http.server` de la librería estándar de Python (se deduce del
header `Server:`, de la página de error 404 y de que un `HEAD` devuelve `501`).

**La espera no es el problema:** con 3 personas y ~10 s por foto, la última
espera 30 s. Tolerable.

**El problema es que el servidor se bloquea entero mientras procesa.** No sirve
el CSS, no responde un export, no entrega la página. Pasa incluso con una sola
persona procesando.

| Problema | ¿Se arregla pagando más en Render? |
|---|---|
| Apagado por inactividad, el `503` | **Sí** |
| Fotos lentas | **Sí, en parte** (más CPU) |
| `run_id` borrado en cada deploy | **Sí**, con disco persistente |
| El servidor se bloquea al procesar | **No.** Es código |
| `http.server` no es apto para producción | **No.** Es código |

**2.1** ¿Usa `HTTPServer` o `ThreadingHTTPServer`? Si es el primero, cambiarlo
por el segundo es casi una línea y elimina el bloqueo. Para 3 usuarios puede ser
suficiente.

**2.2** Medir cuánto tarda y cuánta memoria consume procesar una foto **en el
servidor**.

**2.3** Verificación: 3 personas procesando a la vez, comprobando que una cuarta
que solo abre la página la vea responder normal.

**2.4** Migrar a FastAPI + Uvicorn con 2-3 workers. No urgente a esta escala,
pero la documentación de Python advierte que `http.server` no es para producción
y hay un endpoint público que recibe archivos. Hacerlo antes de difundir la URL.
Ojo: los workers necesitan CPU real.

---

## 3. Bug latente: el `run_id`

`/api/export` recupera la sesión por el `run_id`, y esos archivos viven en el
disco del contenedor, que se borra en cada deploy.

**Cómo se va a manifestar:** alguien procesa una foto, pasa veinte minutos
ajustando puntos y cotas, se publica una versión nueva, aprieta "Exportar DXF" y
falla. Pierde todo el trabajo.

→ Disco persistente o almacenamiento de objetos. Definir cuánto viven los
trabajos y limpiarlos solos. Que el frontend avise claro si el `run_id` ya no
existe.

---

## 4. Precisión: bajar el error de 4 mm

Las mediciones contra molde físico dan diferencias de hasta 4 mm. Razonable para
esta etapa, con margen de mejora.

**4.1 Diagnosticar primero.** Un error de escala es proporcional (4 mm en 300 mm
= 1,3%, o sea ~8 mm en 600 mm); uno de borde es constante. Medir a mano tres
distancias muy distintas (100, 300, 600 mm) sobre el mismo molde y ver cuál se
mantiene estable: el porcentaje o el milimetraje.

El reporte ya trae `recovered_side_mm` — cuánto mide el motor los marcadores
después de rectificar. Si da distinto del tamaño real impreso, ahí está el error
de escala. Reportarlo en cada prueba.

**4.2 `scale_correction`.** El motor ya tiene ese parámetro para compensar una
impresión que no salió al 100%. Si los marcadores se imprimieron al 98,7%, todo
mide 1,3% menos: 4 mm en un molde de 300 mm. → Medir con calibre el lado real de
un marcador impreso. Es la prueba más barata y podría explicar el error entero.

**4.3 Corrección de lente.** `camera.py` implementa `undistort_image()` y
`undistort_points_to_pixels()`, y `stage1.py` y `aruco_stage1.py` las usan.
`marker_metric.py`, que es el pipeline en producción, no las referencia. El
propio código lo lista como limitación: *"Sin corrección de lente: válido sólo si
la distorsión es baja frente al objetivo."*

Como todas las fotos se toman con celular, parece relevante. **¿Fue una decisión
deliberada o quedó pendiente? ¿Qué proponés para mejorar la precisión en este
punto?**

**4.4 Tres hojas A4 imprimibles.** Pedido: que prepare hojas para poder calibrar
y verificar sin depender de él — con marcadores en distintos tamaños para medir
si el tamaño mejora la precisión, y con una figura patrón de medidas exactas para
validar contra algo conocido. Que **todas incluyan una barra de control de 100 mm
impresa**, para verificar con una regla que la impresora no escaló el papel; sin
eso toda medición arrastra el error de impresión. Que documente con qué
configuración imprimirlas.

**4.5 ¿Se puede bajar de 14 marcadores a 6 o 4?** A 6 probablemente sí; a 4 es
riesgoso: sin redundancia, si uno queda tapado por el molde o con brillo la foto
no sirve; y los chequeos de consistencia necesitan marcadores de sobra para
detectar uno mal pegado. Importa más la cobertura del área que la cantidad.

→ Medirlo, no estimarlo: el motor recibe los marcadores por ID, así que se puede
reprocesar **las fotos que ya existen** usando solo 6 y solo 4, y comparar contra
el resultado con 14. Sin reimprimir ni sacar fotos nuevas.

**4.6 Espesor.** Con 1 mm el error por apoyarse encima del plano de los
marcadores es menor a 1 mm: no hace falta compensarlo. Igual conviene documentar
que la foto se tome lo más perpendicular posible.

**4.7 Fijar un objetivo.** Acordar el error aceptable (por ejemplo, menos de
1,5 mm en distancias de hasta 500 mm) y dejar un test que lo verifique. Sin un
número, "mejorar la precisión" no tiene final.

---

## 5. Calidad y seguridad

- **Tests**: al menos uno sobre una foto conocida que verifique que las medidas
  no cambian. Sin eso, cualquier cambio futuro puede degradar la precisión sin
  que nadie se entere.
- **GitHub Actions** en cada PR, y deploy automático al mergear a `main`.
- **Cerrar el endpoint de procesamiento** con clave o login antes de difundir la
  URL: hoy está abierto y consume mucho CPU por pedido.
- **Mensajes de error útiles** y límite de tamaño de archivo con aviso claro.
- **El ejemplo de `docs/PROYECTO.md` no funciona**: pasa una lista donde
  `process_marker_metric` espera una ruta única, y usa `marker_mm` cuando el
  parámetro se llama `marker_size_mm`. Falla con `TypeError`.
- **Descripción en los PRs**: el del motor quedó con el título cortado como cuerpo.
- **Peso del repo**: las 10 fotos suman ~39 MB permanentes en el historial de
  git. ¿Hacen falta todas o alcanzan 2 o 3?

---

## 6. Preguntas

1. ¿Cuánto miden **realmente** los marcadores impresos? El código asume 50 mm y
   de ahí sale toda la escala.
2. ¿Cuánto tarda y cuánta memoria usa procesar una foto en el servidor?
3. ¿`HTTPServer` o `ThreadingHTTPServer`?
4. ¿`moldes.crevia.com.ar` está en una cuenta personal? ¿Con qué plan?
5. ¿Hay condiciones conocidas (luz, ángulo, resolución) donde se pierde precisión?
6. ¿Qué error considerás alcanzable de forma realista?

---

## Prioridades

| # | Qué | Esfuerzo |
|---|---|---|
| 1 | Código de la app al repo + runbook + cuentas | Medio |
| 2 | Medir el marcador impreso y aplicar `scale_correction` | Minutos |
| 3 | Confirmar `HTTPServer` vs `ThreadingHTTPServer` | Minutos |
| 4 | Medir tiempo y memoria de procesamiento | Minutos |
| 5 | Propuesta para la corrección de lente | A definir |
| 6 | Persistencia del `run_id` | Bajo |
| 7 | Hojas A4 de calibración y validación | Bajo |
| 8 | Probar 6 y 4 marcadores sobre las fotos existentes | Bajo |
| 9 | Tests y CI | Medio |
| 10 | Migrar a FastAPI + Uvicorn | Medio |

Los puntos 2, 3 y 4 cuestan minutos y definen casi todo lo demás.

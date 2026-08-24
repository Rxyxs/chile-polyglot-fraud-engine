[ 🇺🇸 Read in English ](README.md) | [ 🇨🇱 Español ]

# 1. Nombre del Proyecto

## Motor Poliglota de Fraude — Nucleo de Velocidad en C + DSL de Reglas en Ruby + ML en Python

![C](https://img.shields.io/badge/C-C11-A8B9CC?style=flat&logo=c&logoColor=white)
![Ruby](https://img.shields.io/badge/Ruby-3.2%2B-CC342D?style=flat&logo=ruby&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.x-02569B?style=flat)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.11x-009688?style=flat&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.3x-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Tests](https://img.shields.io/badge/tests-45%20passing%20(C%2BRuby%2BPython)-brightgreen?style=flat)
![Status](https://img.shields.io/badge/status-research%20%2F%20datos%20sinteticos-lightgrey?style=flat)

Un motor de deteccion de fraude en tres lenguajes para transacciones
bancarias chilenas, donde cada lenguaje hace el trabajo para el que
realmente es mejor: **C** calcula metricas de velocidad/geolocalizacion de
forma nativa en nanosegundos, **Ruby** evalua un DSL de reglas de negocio
legible (listas negras, paises de alto riesgo, patrones de estructuracion),
y **Python** entrena y sirve un ensemble IsolationForest + LightGBM que
consume las salidas de las otras dos capas como atributos. Un solo comando
(`make all`) compila, genera los datos y entrena todo; `make test` ejecuta
los 45 tests (11 en C, 10 RSpec, 24 pytest) — cada numero de este README
proviene de ejecutar eso en esta maquina.

---

# 2. Motivacion

La mayoria de los proyectos de portafolio de "deteccion de fraude" son un
solo notebook de Python con un clasificador. Eso no refleja como se
construye el stack de fraude de un banco real: lenguajes y servicios
distintos, cada uno elegido porque es la herramienta correcta para un
trabajo especifico, conectados con restricciones de interoperabilidad
reales. Construi este proyecto para trabajar ese problema de integracion
directamente, no para simularlo:

1. **La aritmetica del hot-path (distancia haversine, velocidad de viaje
   implicita, z-score de monto) tiene que ser genuinamente rapida**, porque
   corre en cada transaccion, no solo en los datos de entrenamiento. C
   compilado a una biblioteca nativa compartida y llamado via `ctypes` es
   el ajuste natural — y quise *probar* la afirmacion de sub-milisegundo en
   vez de solo afirmarla, asi que `src/c/bench_main.c` mide la funcion
   compilada de forma aislada (2.000.000 de llamadas) como parte del build.
2. **Las reglas de negocio (listas negras, umbrales de patrones
   normativos) cambian seguido y las escriben analistas de riesgo, no
   ingenieros de ML.** Un pequeño DSL interno es la forma correcta para
   eso — la sintaxis de bloques de Ruby hace que
   `rule "monto_excesivo", weight: 40 do |txn| ... end` se lea casi como
   prosa de politica, que es justamente el punto de construir las reglas
   como un DSL en vez de una cadena de `if` enterrada en el codigo de
   scoring.
3. **Mantener tres lenguajes en una ruta de solicitud de baja latencia es
   en si mismo el problema de sistemas dificil.** Un diseño ingenuo
   ejecutaria `ruby rules_engine.rb` una vez por transaccion, pagando el
   costo de arranque del interprete de Ruby cada vez. Medi esa diferencia
   directamente (ver §6) y construi en su lugar un puente de subproceso
   persistente, porque la diferencia entre "correcto" y "correcto Y lo
   suficientemente rapido para producción" es el contenido de ingenieria
   real aqui, no una nota al pie.

# 3. Arquitectura

```
Transaccion
   |
   v
src/python/api.py -- FastAPI /detect-fraud
   |
   |---> [C]    src/c/fraud_core.c
   |            llamado en el mismo proceso via ctypes, ~2 microseg/llamada
   |            -> distancia haversine, velocidad de viaje imposible,
   |               z-score de monto, velocity_score
   |
   |---> [Ruby] src/ruby/rules_engine.rb
   |            subproceso `--server` persistente, JSON por linea
   |            sobre stdin/stdout, ~0,07ms/llamada
   |            -> comercio en lista negra, pais de alto riesgo,
   |               patron de estructuracion, regla de rafaga de velocidad
   |
   v
[Python] src/python/train_model.py
         IsolationForest (50 arboles, pre-filtro de anomalias)
         -> LightGBM (clasificador final)

         Conjunto de atributos = atributos crudos de la transaccion
                      + velocity_score de C
                      + rules_risk_score / rules_flagged de Ruby
                      + puntaje de anomalia propio de IsolationForest

         La capa de ML aprende a PONDERAR las salidas de las otras
         dos capas, no solo a votar junto a ellas.
   |
   v
fraud_probability, is_fraud
(+ desglose completo: c_layer, ruby_layer -- ver FraudDetectionResponse)
```

`src/app/dashboard.py` (Streamlit) reproduce el split de prueba retenido,
desglosando que capa realmente detecto cada arquetipo de fraude.

# 4. Por Que Cuatro Arquetipos de Fraude, No Uno

`src/python/generate_data.py` inyecta cuatro patrones de fraude
*distintos*, deliberadamente, para que ninguna capa sola resuelva todo el
problema — demostrando que el diseño en capas tiene una razon genuina de
existir:

| Arquetipo | Señal | Capa que realmente lo detecta |
|---|---|---|
| `velocity_burst` | Transacciones rapidas + un salto geografico | C (`velocity_score`, `is_impossible_travel`) |
| `blacklisted_merchant` | Una sola transaccion en un comercio conocido como malo, por lo demas normal | Ruby (regla `comercio_en_lista_negra`) |
| `high_risk_country` | Transaccion marcada con un codigo de pais de alto riesgo | Ruby (regla `pais_alto_riesgo`) |
| `structuring` | Varias transacciones justo bajo el umbral de reporte de UF 450 en un dia | Ruby (regla `estructuracion_subumbral`) |

Las filas de `blacklisted_merchant`/`high_risk_country`/`structuring` no
llevan ninguna anomalia de velocidad o geolocalizacion — la capa C no ve
nada inusual en ellas. LightGBM es la capa que finalmente combina todo esto
en una sola probabilidad.

**Descargo especifico de Chile**: el umbral de reporte en efectivo de UF
450 (`UF_REPORT_THRESHOLD_CLP` en `src/ruby/rules_engine.rb`) es una
aproximacion ilustrativa inspirada en normas AML/CMF chilenas (Ley 19.913),
no una cifra legal verificada, y `UF_TO_CLP` es una conversion ilustrativa
fija, no un valor indexado en vivo. No se usa ninguna lista negra,
comercio, ni dato regulatorio real en ningun lugar de este proyecto — ver
§9.

# 5. Un Bug Real Encontrado y Corregido al Validar Esto

`BLACKLISTED_MERCHANT_IDS` originalmente incluia `MER_00013` — que,
resulto, caia *dentro* del rango de IDs del pool de comercios legitimos
(`MER_00001`..`MER_00199`, ver `N_MERCHANTS` en `generate_data.py`).
Transacciones legitimas ordinarias podian caer en `MER_00013` por pura
casualidad, y 230 de ellas lo hicieron, contra solo 72 filas de fraude
usando el mismo ID — contaminando la señal de "comercio en lista negra"
con ruido real en las etiquetas.

Esto se detecto empiricamente, no por revision de codigo: el F1 general
del primer modelo entrenado fue 0,831, y al desglosar el recall por
`fraud_type` (ver la pestaña "Contribucion por capa" de
`src/app/dashboard.py`, construida exactamente para este tipo de
diagnostico) el recall de `blacklisted_merchant` era de solo 44%, mientras
los otros tres arquetipos ya estaban al 100%. Mover el ID en colision a
`MER_00777` (fuera del rango legitimo, como los otros dos IDs en lista
negra) y reentrenar llevo el recall de `blacklisted_merchant` a 100% y el
F1 general a 0,995 — ver
`tests/python/test_generate_data.py::test_blacklisted_merchant_ids_never_collide_with_legit_pool`
para el test de regresion en que se convirtio esto.

# 6. Resultados (Numeros Reales de una Ejecucion Real)

`make all` en esta maquina (semilla 42, reproducible desde un clon
limpio): 50.000 transacciones sinteticas, tasa de fraude de 1,512%
(756 filas fraudulentas entre los cuatro arquetipos), 3.000 clientes,
split temporal (train 35.000 / val 7.500 / test 7.500, cronologico, sin
mezclar).

| Metrica | Valor (split de prueba) |
|---|---|
| Precision | 0,982 |
| Recall | **1,000** (110/110 fraudes detectados) |
| F1 | 0,991 |
| ROC-AUC | 1,000 |
| PR-AUC | 0,99999... |
| Falsos positivos | 2 (de 7.390 transacciones legitimas) |

Recall por arquetipo de fraude en el split de prueba, despues de la
correccion del §5: `velocity_burst` 35/35, `blacklisted_merchant` 43/43,
`high_risk_country` 16/16, `structuring` 16/16 — todos los arquetipos al
100% ahora.

## Latencia (medida, no estimada — `tests/python/test_latency.py`)

| Capa | Costo medido | Como |
|---|---|---|
| Modulo C, benchmark en C puro | **45,5 ns/llamada** | `src/c/bench_main.c`, 2.000.000 iteraciones, `make bench-c` |
| Modulo C via `ctypes` desde Python | p50 0,0020ms, p95 0,0021ms | 2.000 llamadas, en el mismo proceso |
| Ida y vuelta del motor de reglas Ruby (subproceso persistente) | p50 0,070ms, p95 0,092ms, max 0,237ms | 300 llamadas sobre JSON por stdin/stdout |
| `score_samples()` de IsolationForest, una fila | ~1,8ms | costo dominante en el pipeline completo — ver abajo |
| `predict()` de LightGBM, una fila | p50 0,064ms, p95 0,112ms | |
| **Solicitud completa `/detect-fraud`** (todas las capas + FastAPI) | **p50 4,22ms, p95 4,80ms, max 5,58ms** | 200 solicitudes via `TestClient` de FastAPI |

**Hallazgo honesto**: el objetivo de "< 1ms" del brief es real y se cumple
— para el modulo C especificamente, a 45,5 nanosegundos por llamada, cuatro
ordenes de magnitud bajo el presupuesto. *No* es lo que logra el pipeline
completo de punta a punta, y afirmar lo contrario tergiversaria la
medicion. El cuello de botella real es `IsolationForest.score_samples()`
de `sklearn`: su costo por llamada esta dominado por la iteracion a nivel
de Python sobre los arboles, que apenas se amortiza para una prediccion de
una sola fila (el caso comun en produccion, a diferencia del entrenamiento).
Escala aproximadamente lineal con `n_estimators` — medido directamente: 200
arboles cuestan ~6,6ms/llamada, 50 arboles cuestan ~1,8ms/llamada, mientras
que el `predict()` de una sola fila de LightGBM se mantiene bajo 0,1ms sin
importar la cantidad de arboles. `train_model.py` usa 50 estimadores
especificamente por esta medicion (ver el comentario sobre
`IsolationForest(...)` en `src/python/train_model.py`), reduciendo la
latencia p50 del pipeline completo de 12,1ms a 4,2ms sin costo medible en
recall/precision (confirmado reentrenando y reevaluando en ambas
configuraciones). Un despliegue de produccion que persiga latencia de
punta a punta sub-milisegundo reemplazaria o agruparia (batch) este paso;
este repositorio reporta el trade-off en vez de ocultarlo.

![Curva Precision-Recall](outputs/plots/precision_recall_curve.png)
![Matriz de Confusion](outputs/plots/confusion_matrix.png)
![Importancia de Atributos](outputs/plots/feature_importance.png)

# 7. Estructura del Repositorio

```
chile-polyglot-fraud-engine/
├── data/
│   ├── raw/                    # transactions.parquet/.csv generado (en .gitignore)
│   └── processed/              # features.parquet de ingenieria de atributos (en .gitignore)
├── src/
│   ├── c/
│   │   ├── fraud_core.h/.c     # metricas de velocidad/geo, compilado a fraud_core.dll
│   │   ├── bench_main.c        # valida < 1ms/llamada
│   │   ├── build.ps1           # build con MSVC (vcvars64 + cl)
│   │   └── Makefile            # build | test | bench | clean
│   ├── ruby/
│   │   ├── rules_engine.rb     # DSL RuleSet + modo --server stdin/stdout
│   │   └── Gemfile
│   ├── python/
│   │   ├── generate_data.py    # transacciones bancarias chilenas sinteticas, 4 arquetipos de fraude
│   │   ├── bridge.py            # puente ctypes (C) + puente de subproceso persistente (Ruby)
│   │   ├── train_model.py       # construccion de atributos + IsolationForest + LightGBM
│   │   └── api.py               # FastAPI /detect-fraud, combina las 3 capas
│   └── app/dashboard.py         # Streamlit: atribucion por capa, replay en vivo, mapa
├── tests/
│   ├── c/test_fraud_core.c      # harness basado en asserts, compilado por src/c/build.ps1
│   ├── ruby/rules_engine_spec.rb
│   └── python/                  # bridge, generate_data, train_model, api, latencia
├── outputs/
│   ├── models/       # DLL/exe compilados + artefactos entrenados (en .gitignore, `make all` regenera)
│   ├── plots/        # curva PR, matriz de confusion, importancia de atributos (versionados)
│   └── reports/      # training_report.json (en .gitignore, los numeros estan en este README)
├── Makefile
├── requirements.txt
├── pytest.ini
├── README.md
└── README.es.md
```

# 8. Instalacion y Uso

Requiere: **MSVC** (Visual Studio Build Tools o Visual Studio con el
workload "Desktop development with C++") para la capa C; **Ruby 3.2+** con
Bundler; **Python 3.10+** (el codigo usa sintaxis de union PEP 604
`str | None` de forma nativa, por lo que 3.10 es un piso real); **GNU
Make** (en Windows, ej. `winget install ezwinports.make`).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Compila la biblioteca C, instala gemas Ruby, genera datos y entrena todo
make all

# Ejecuta los 45 tests en los tres lenguajes
make test

# Levanta la API de scoring en tiempo real (combina C + Ruby + Python por solicitud)
make run-api
# luego: POST http://localhost:8000/detect-fraud

# Levanta el dashboard de monitoreo
make run-dashboard
```

Targets individuales: `make build-c`, `make test-c`, `make bench-c`,
`make install-ruby`, `make test-ruby`, `make generate-data`, `make train`,
`make test-python`, `make clean`. Ejecuta `make help` para la lista
completa.

# 9. Descargo de Responsabilidad

Todos los datos de transacciones son generados sinteticamente
(`src/python/generate_data.py`, con semilla fija, reproducible) con fines
de demostracion. No se utilizan datos bancarios reales, datos de clientes,
listas negras de comercios, ni logica de deteccion de fraude propietaria de
ninguna institucion financiera. Los umbrales inspirados en normas AML/CMF
chilenas del motor de reglas Ruby (§4) son aproximaciones ilustrativas para
una demostracion con datos sinteticos, no cifras legales verificadas —
consulta las fuentes oficiales de la CMF/UAF para umbrales de cumplimiento
reales.

# 10. Licencia

MIT — ver [LICENSE](LICENSE) para el texto completo.

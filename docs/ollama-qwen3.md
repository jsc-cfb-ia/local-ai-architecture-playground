# Ollama y Qwen3 4B

## Por qué este modelo

`qwen3:4b` es un paso intermedio apropiado para este laboratorio:

- tiene 4.02 mil millones de parámetros;
- la variante publicada en Ollama usa cuantización `Q4_K_M`;
- la descarga del modelo ocupa aproximadamente 2.5 GB;
- soporta varios idiomas y casos de uso con herramientas;
- cabe con margen en un MacBook Air con 24 GB de memoria unificada.

El tamaño del archivo no representa toda la memoria usada durante inferencia:
también se necesita espacio para el contexto, cachés y el runtime.

## Instalación en macOS

Descargar Ollama desde:

<https://ollama.com/download>

Comprobar la instalación:

```bash
ollama --version
```

Descargar el modelo:

```bash
ollama pull qwen3:4b
```

Comprobarlo directamente:

```bash
ollama run qwen3:4b
```

Salir del chat interactivo con `/bye`.

## Ejecutar el playground

Desde la raíz del proyecto:

```bash
LOCAL_AI_PROVIDER=ollama \
OLLAMA_MODEL=qwen3:4b \
python -m app.main
```

Dentro del asistente:

```text
You: status
Provider: ollama
Model: qwen3:4b
Endpoint: http://localhost:11434
Fallback: lexical retrieval
```

El comando `status` muestra configuración, no garantiza que el proceso Ollama
esté respondiendo. Una pregunta real prueba la conexión.

## Probar la API local

```bash
curl http://localhost:11434/api/chat \
  -d '{
    "model": "qwen3:4b",
    "messages": [
      {
        "role": "user",
        "content": "Explain event-driven architecture briefly."
      }
    ],
    "stream": false,
    "think": false
  }'
```

## Parámetros del proyecto

- `stream: false`: simplifica la primera integración.
- `think: false` y `/no_think`: solicitan el modo de respuesta directa.
- `temperature: 0.2`: busca consistencia técnica.
- `keep_alive: 5m`: mantiene el modelo cargado entre preguntas cercanas.
- timeout: 120 segundos.

## Baseline local inicial

Medición observada durante la primera integración en el MacBook Air con 24 GB:

- modelo cargado: `qwen3:4b`;
- tamaño reportado en ejecución: 3.2 GB;
- procesador reportado por Ollama: 100% GPU;
- contexto reportado: 4096 tokens;
- respuesta RAG observada: aproximadamente 15 a 23 segundos.

Estos valores son una referencia de desarrollo, no un benchmark. Pueden variar
según longitud del prompt, longitud de la respuesta, temperatura del equipo,
aplicaciones abiertas y si el modelo ya está cargado.

Consultar el estado actual:

```bash
ollama ps
```

## Diagnóstico

### `command not found: ollama`

Ollama no está instalado o no está disponible en `PATH`.

### Modelo no encontrado

Ejecutar:

```bash
ollama pull qwen3:4b
ollama list
```

### Conexión rechazada

Abrir la aplicación Ollama o iniciar su servicio. La API esperada es:

```text
http://localhost:11434
```

### Primera respuesta lenta

La primera llamada carga el modelo en memoria. Las llamadas posteriores suelen
beneficiarse de `keep_alive`.

## Referencias

- Modelo: <https://ollama.com/library/qwen3:4b>
- Inicio rápido: <https://docs.ollama.com/quickstart>
- Chat API: <https://docs.ollama.com/api/chat>

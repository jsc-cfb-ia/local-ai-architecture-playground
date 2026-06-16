# Demo 8: Local Semantic Retrieval

## Objetivo

Demo 8 introduce embeddings locales para comparar búsqueda lexical y búsqueda
semántica sin usar APIs externas. La aplicación sigue siendo local-first:

- Ollama ejecuta el modelo de embeddings;
- los documentos siguen viviendo en `knowledge/`;
- la cache de vectores vive en `.cache/embeddings/`;
- Git ignora la cache porque puede reconstruirse desde los documentos fuente.

## Modelo elegido

El modelo inicial es `nomic-embed-text`.

Motivos:

- pesa aproximadamente 274 MB;
- es un modelo dedicado a embeddings;
- Ollama lo expone mediante `/api/embed`;
- es liviano para el MacBook Air con 24 GB.

Instalar:

```bash
ollama pull nomic-embed-text
```

## API de embeddings

Ollama genera embeddings con:

```bash
curl http://localhost:11434/api/embed \
  -d '{
    "model": "nomic-embed-text",
    "input": "Why is the sky blue?"
  }'
```

El endpoint devuelve una lista de vectores numéricos. La aplicación calcula
similitud coseno entre el vector de la pregunta y los vectores de los chunks.

## Componentes

### `app/embeddings.py`

Define el contrato `EmbeddingModel` y el adaptador `OllamaEmbeddingModel`.
Este contrato permite cambiar de runtime o modelo sin tocar el retriever
semántico.

### `app/semantic_retriever.py`

Responsable de:

- leer chunks existentes;
- calcular o recuperar embeddings cacheados;
- generar el embedding de la pregunta;
- ordenar chunks por similitud coseno.

### `.cache/embeddings/`

La cache se organiza por modelo. Cada entrada usa una clave estable basada en
fuente, topic, número de chunk y hash del contenido. Si el documento cambia,
el hash cambia y el embedding se recalcula.

## Ejecutar evaluación

Lexical:

```bash
python -m app.evaluate --strategy lexical
```

Semántica:

```bash
python -m app.evaluate --strategy semantic
```

Comparación:

```bash
python -m app.evaluate --strategy both
```

Resultado observado al iniciar Demo 8:

```text
lexical:  10/10 passed
semantic: 10/10 passed
```

## Qué aprendemos

La búsqueda lexical responde bien cuando la pregunta comparte términos con los
documentos. La búsqueda semántica puede recuperar chunks relacionados aunque
la pregunta use vocabulario diferente, pero introduce nuevas preocupaciones:

- elección del modelo de embeddings;
- costo de indexar documentos;
- cache e invalidación;
- métricas para comparar ranking;
- explicación de por qué un chunk fue recuperado.

## Límites actuales

- El asistente principal aún usa retrieval lexical.
- No existe todavía búsqueda híbrida.
- El dataset sigue siendo pequeño.
- La evaluación mide fuente top-1 y términos esperados, no fidelidad completa
  de la respuesta generada.

## Próximo paso

Demo 8 debe cerrar con una decisión explícita:

1. mantener lexical como default;
2. cambiar a semantic;
3. crear un ranking híbrido que combine exactitud lexical y similitud semántica.

La opción más probable será híbrida, porque nombres de servicios como SQS,
AppSync y Step Functions se benefician de coincidencia exacta, mientras que
preguntas conceptuales se benefician de embeddings.

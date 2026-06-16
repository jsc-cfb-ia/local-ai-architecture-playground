# Evaluación de retrieval

## Objetivo

La evaluación comprueba si el retriever coloca una fuente esperada en la
primera posición y si el chunk superior contiene términos mínimos esperados.
No evalúa todavía la calidad lingüística de Qwen ni la fidelidad frase por
frase de la respuesta.

El dataset vive en:

```text
evaluation/questions.json
```

Cada caso contiene:

- identificador estable;
- pregunta;
- una o más fuentes aceptables.
- términos que deben aparecer en el chunk recuperado.

## Ejecutar

```bash
python -m app.evaluate
```

Comparar lexical y semántico:

```bash
python -m app.evaluate --strategy both
```

El comando imprime el resultado de cada pregunta y termina con código distinto
de cero si algún caso falla.

Resultado esperado para Demo 7:

```text
Summary: 10/10 passed (100% top-1 source accuracy)
```

Demo 8 agrega `--strategy semantic`, que utiliza embeddings locales de Ollama.
La primera ejecución puede tardar más porque construye `.cache/embeddings/`.

## Preguntas cubiertas

La evaluación incluye:

- definición de GraphQL;
- operaciones de AppSync;
- definición de Step Functions;
- servicios event-driven;
- buffering con SQS;
- idempotencia;
- dead-letter queues;
- retries;
- estado de pedidos en DynamoDB;
- identificadores de observabilidad.

## Cómo evolucionarla

Demo 8 reutiliza estas preguntas para comparar:

1. ranking lexical actual;
2. búsqueda por embeddings;
3. retrieval híbrido;
4. cambios de chunking.

Cuando el dataset crezca, conviene separar métricas:

- top-1 source accuracy;
- recall dentro del top 3;
- preguntas sin respuesta local;
- relevancia manual del chunk;
- fidelidad de la respuesta generada.

Las preguntas deben añadirse antes de ajustar el algoritmo para reducir el
riesgo de diseñar una evaluación que solo confirme el comportamiento existente.

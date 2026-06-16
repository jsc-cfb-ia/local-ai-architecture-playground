# Ruta de evolución

## Principio

El playground debe evolucionar mediante contratos pequeños. El retriever, el
modelo y el consumidor no deben conocerse más de lo necesario. Así una mejora
interna no rompe las skills o agentes que utilicen la capacidad.

## Etapa 1: RAG local observable

Estado actual:

- documentos locales;
- chunking;
- ranking lexical;
- contexto reciente;
- síntesis opcional con Qwen3 4B;
- fuentes y fallback;
- pruebas unitarias.

Siguiente mejora recomendada:

- crear un conjunto de 15 a 30 preguntas esperadas;
- registrar fuentes esperadas;
- comparar precisión antes de cambiar el retriever.

## Etapa 2: Herramienta local estable

Crear una función pública con entrada y salida serializables:

```json
{
  "question": "How should Step Functions integrate with SQS?",
  "context": "The workflow is asynchronous."
}
```

```json
{
  "content": "...",
  "sources": ["step-functions.txt"],
  "generated_by_model": true,
  "warning": ""
}
```

Esta será la frontera que consumirán una API, MCP o una skill.

## Etapa 3: MCP

Exponer una herramienta como:

```text
ask_architecture(question, context?)
```

Responsabilidades del servidor MCP:

- validar entrada;
- llamar a `ArchitectureAssistant.answer()`;
- devolver datos estructurados;
- no duplicar retrieval ni prompts.

Un agente externo decidirá cuándo invocar la herramienta. El Architecture
Assistant seguirá controlando cómo busca y fundamenta la respuesta.

## Etapa 4: Skills de arquitectura

Construir capacidades específicas sobre el mismo núcleo:

- revisar un diseño event-driven;
- proponer una máquina de estados;
- generar un ADR;
- describir un sistema para IcePanel;
- revisar un esquema GraphQL/AppSync;
- producir un sequence flow.

Cada skill debe definir:

- propósito;
- entrada estructurada;
- pasos;
- herramientas permitidas;
- salida verificable;
- criterios para rechazar o pedir información.

## Etapa 5: Retrieval híbrido

Demo 8 empieza esta etapa agregando embeddings locales sin eliminar el ranking
lexical:

1. búsqueda lexical para términos exactos como nombres de servicios;
2. búsqueda vectorial para similitud semántica;
3. combinación y reranking;
4. evaluación contra el conjunto de preguntas.

La caché de embeddings debe vivir localmente y poder reconstruirse desde los
documentos fuente.

## Etapa 6: Flujo multiagente

Introducir agentes solo cuando existan herramientas estables y evaluadas.
Posibles roles:

- analista de requisitos;
- diseñador de arquitectura;
- revisor de seguridad y resiliencia;
- generador de documentación;
- crítico que busca supuestos y contradicciones.

El objetivo no debe ser aumentar el número de agentes, sino separar tareas con
entradas, salidas y criterios de terminación claros.

## Métricas de evolución

- precisión de la fuente recuperada;
- respuestas sin soporte documental;
- latencia total y de generación;
- tasa de fallback;
- tamaño del prompt;
- consistencia entre ejecuciones;
- utilidad de la salida para la siguiente herramienta o agente.

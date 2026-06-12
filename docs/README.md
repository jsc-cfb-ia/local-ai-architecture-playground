# Documentación del Architecture Assistant

Este proyecto es un laboratorio local para aprender a construir aplicaciones
de IA de manera incremental, observable y sin depender de APIs de pago.

El objetivo no es crear solamente un chatbot. Estamos construyendo un núcleo
de asistencia arquitectónica que pueda ser consumido mediante diferentes
interfaces:

- terminal interactiva;
- API HTTP local;
- herramienta para otro agente;
- servidor MCP;
- skill especializada;
- flujo multiagente.

## Documentos

- [Arquitectura](architecture.md): componentes, flujo RAG y contratos.
- [Ollama y Qwen3 4B](ollama-qwen3.md): instalación, ejecución y diagnóstico.
- [Guía de demo](demo-guide.md): preparación, guion y plan de contingencia.
- [Evaluación](evaluation.md): dataset de preguntas y métrica de retrieval.
- [Evolución](evolution.md): etapas para convertir el playground en una
  plataforma de herramientas y agentes.

## Estado actual

El asistente dispone de dos modos:

1. `retrieval`: recupera y muestra conocimiento local sin usar un LLM.
2. `ollama`: recupera conocimiento y solicita a `qwen3:4b` que redacte una
   respuesta fundamentada en esos fragmentos.

Si Ollama no está disponible, el modo `ollama` vuelve automáticamente a la
respuesta de retrieval. Esto mantiene el sistema funcional y facilita el
diagnóstico.

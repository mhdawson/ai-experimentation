import * as traceloop from "@traceloop/node-server-sdk";
import { trace, context } from "@opentelemetry/api";

import * as LlamaIndex from "llamaindex";

import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-grpc";
import { ConsoleSpanExporter } from "@opentelemetry/sdk-trace-node";

traceloop.initialize({ exporter: new OTLPTraceExporter(),
                        disableBatch: true,
                        instrumentModules: {
                          llamaIndex: LlamaIndex,
                        }
                      });

const tracer = trace.getTracer();

import {askQuestions} from "./llamaindex-function-ollama.mjs"
tracer.startActiveSpan('Asking questions', async (span) => {
  await askQuestions();
  span.end();
});

setInterval(() => {
  console.log('alive');
}, 5000);

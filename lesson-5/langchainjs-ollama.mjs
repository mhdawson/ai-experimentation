import { ChatPromptTemplate } from "@langchain/core/prompts";
import path from "path";
import {fileURLToPath} from "url";

////////////////////////////////
// GET THE MODEL
const model = await getModel('ollama', 0.9);
//const model = await getModel('llama-cpp', 0.9);
//const model = await getModel('openAI', 0.9);
//const model = await getModel('Openshift.ai', 0.9);

////////////////////////////////
// CREATE CHAIN
const prompt =
  ChatPromptTemplate.fromTemplate(`Answer the following question if you don't know the answer say so:

Question: {input}`);

const chain = prompt.pipe(model);

////////////////////////////////
// ASK QUESTION
console.log(new Date());
let result = await chain.invoke({
  input: "Should I use npm to start a node.js application",
});
console.log(result);
console.log(new Date());

/////////////////////////////////////////////////
// HELPER FUNCTIONS
async function getModel(type, temperature) {
  console.log("Loading model - " + new Date());

  let model;
  if (type === 'llama-cpp') {
    ////////////////////////////////
    // LOAD MODEL
    const __dirname = path.dirname(fileURLToPath(import.meta.url));
    const modelPath = path.join(__dirname, "models", "mistral-7b-instruct-v0.1.Q5_K_M.gguf")
    const { LlamaCpp } = await import("@langchain/community/llms/llama_cpp");
    model = await new LlamaCpp({ modelPath: modelPath,
                                 batchSize: 1024,
                                 temperature: temperature,
                                 gpuLayers: 64 });
  } else if (type === 'openAI') {
    ////////////////////////////////
    // Connect to OpenAPI
    const { ChatOpenAI } = await import("@langchain/openai");
    const key = await import('../key.json', { with: { type: 'json' } });
    model = new ChatOpenAI({
      temperature: temperature,
      openAIApiKey: key.default.apiKey
    });
  } else if (type === 'Openshift.ai') {
    ////////////////////////////////
    // Connect to OpenShift.ai endpoint
    const { ChatOpenAI } = await import("@langchain/openai");
    model = new ChatOpenAI(
      { temperature: temperature,
        openAIApiKey: 'EMPTY',
        modelName: 'mistralai/Mistral-7B-Instruct-v0.2' },
      { baseURL: 'http://vllm.llm-hosting.svc.cluster.local:8000/v1' }
    );
  } else if (type === 'ollama') {
    ////////////////////////////////
    // Connect to ollama endpoint
    const { Ollama } = await import("@langchain/community/llms/ollama");
    model = new Ollama({
      baseUrl: "http://10.1.1.39:11434", // Default value
      model: "mistral", // Default value
    });
  };

  return model;
};

import {fileURLToPath} from "url";
import path from "path";

import { TextLoader } from "langchain/document_loaders/fs/text";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { MarkdownTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";

import { createRetrievalChain } from "langchain/chains/retrieval";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";


////////////////////////////////
// LOAD AUGMENTING DATA
// typically this is stored in a database versus being loaded every time

console.log("Loading and processing augmenting data - " + new Date());

const docLoader = new DirectoryLoader(
  "./SOURCE_DOCUMENTS",
  {
    ".md": (path) => new TextLoader(path),
  }
);
const docs = await docLoader.load();

const splitter = await new MarkdownTextSplitter({
  chunkSize: 500,
  chunkOverlap: 50
});
const splitDocs = await splitter.splitDocuments(docs);

const vectorStore = await MemoryVectorStore.fromDocuments(
  splitDocs,
  new HuggingFaceTransformersEmbeddings()
);
const retriever = await vectorStore.asRetriever();
console.log("Augmenting data loaded - " + new Date());

////////////////////////////////
// GET THE MODEL
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const modelPath = path.join(__dirname, "models", "mistral-7b-instruct-v0.1.Q5_K_M.gguf")
const { LlamaCpp } = await import("@langchain/community/llms/llama_cpp");
const model = await new LlamaCpp({ modelPath: modelPath,
                                   batchSize: 1024,
                                    gpuLayers: 64 });


////////////////////////////////
// CREATE CHAIN

const prompt =
  ChatPromptTemplate.fromTemplate(`Answer the following question based only on the provided context, if you don't know the answer say so:

<context>
{context}
</context>

Question: {input}`);

const documentChain = await createStuffDocumentsChain({
  llm: model,
  prompt,
});

const retrievalChain = await createRetrievalChain({
  combineDocsChain: documentChain,
  retriever,
});

////////////////////////////////
// ASK QUESTIONS

console.log(new Date());
let result = await retrievalChain.invoke({
  input: "Should I use npm to start a node.js application",
});
console.log(result);
console.log(new Date());

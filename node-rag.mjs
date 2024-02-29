import {fileURLToPath} from "url";
import path from "path";

import { TextLoader } from "langchain/document_loaders/fs/text";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { MarkdownTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

import { LlamaCpp } from "@langchain/community/llms/llama_cpp";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";

import { createRetrievalChain } from "langchain/chains/retrieval";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const modelPath = path.join(__dirname, "models", "mistral-7b-instruct-v0.1.Q5_K_M.gguf")
//const modelPath = path.join(__dirname, "models", "llama-2-7b-chat.Q4_K_M.gguf");


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
  new HuggingFaceTransformersEmbeddings({
    modelPath: modelPath 
  })
);
const retriever = await vectorStore.asRetriever();

console.log("Augmenting data loaded - " + new Date());


////////////////////////////////
// LOAD MODEL

console.log("Loading model - " + new Date());

const model = await new LlamaCpp({ modelPath: modelPath,
                                   gpuLayers: 64 });

console.log("Model loaded - " + new Date());

////////////////////////////////
// CREATE CHAIN and ask questions

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

console.log(new Date());

let result = await retrievalChain.invoke({
  input: "Should I use npm to start a node.js application",
});

console.log(result);

console.log(new Date());

result = await retrievalChain.invoke({
  input: "How do I build a good container for a Node.js application",
});

console.log(result);

console.log(new Date());

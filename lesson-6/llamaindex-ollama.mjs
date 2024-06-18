import {
    Ollama,
    SimpleChatEngine,
} from "llamaindex"

////////////////////////////////
// GET THE MODEL
const llm = new Ollama({
    config: { host: "http://10.1.1.39:11434" },
    model: "mistral", // Default value
});

////////////////////////////////
// CREATE THE ENGINE
const chatEngine = new SimpleChatEngine({ llm });

////////////////////////////////
// ASK QUESTION
const input = 'should I use npm to start a Node.js application';
console.log(new Date());
const response = await chatEngine.chat({
  message: `Answer the following question if you don't know the answer say so: Question: ${input}`
});
console.log(response.response);
console.log(new Date());

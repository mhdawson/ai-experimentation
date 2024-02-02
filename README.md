# Summary

How to use

* Create a directory called SOURCE_FILES and copy markdown documents into it with
  your additional context which should be used to answer questions.
* Create a directory called models and copy the model to use into that directory.
  Update the line which loads the model to point to the model. Models
  can be downloaded from hugging face, for example - 
  [TheBloke/Mistral-7B-Instruct-v0.1-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF)
* run `npm install`
* Edit the line with the question you want to ask.
* run `node node-rag.mjs`


import {
  Ollama,
  SimpleChatEngine,
} from 'llamaindex'

import { inspect } from 'node:util'

////////////////////////////////
// GET THE MODEL
const llm = new Ollama({
    config: { host: 'http://10.1.1.39:11434' },
    model: 'mistral', // Default value
});

////////////////////////////////
// CREATE THE ENGINE
const chatEngine = new SimpleChatEngine({ llm });

////////////////////////////////
// PROMPT
function getQuery(query) {
const input = `
You are a helpful research assistant but do not mention that. The following functions
are available for you to fetch further data to answer user questions, if relevant:

[{
    "function": "favoriteColorTool",
    "description": "returns the favorite color for person given their City and Country",
    "arguments": [
        {
            "name": "city",
            "type": "string",
            "description": "the city for the person"
        },
        {
            "name": "country",
            "type": "string",
            "description": "the country for the person"
        }
    ]
}]

To call a function respond - immediately and only - with a JSON object of the following format:
{
    "function": "function_name",
    "arguments": {
        "argument1": "argument_value",
        "argument2": "argument_value"
    }
}

Only use one of these tools if it is relevant to the question. 

When using the favoriteColorTool if you do not have the users city and Country
ask for it first. Do not guess the users city.

Do not not mention any tools

Do not show JSON when asking user for a city or country


${query}`
return { message: input }; 
}

/////////////////////////////
// FUNCTION IMPLEMENTATION
function getFavoriteColor(city, country) {
  if ((city === 'Ottawa') && (country === 'Canada')) {
    return 'the favoriteColorTool returned black';
  } else if ((city === 'Montreal') && (country === 'Canada')) {
    return 'the favoriteColorTool returned red';
  } else {
    return `the favoriteColorTool returned The city or country
            was not valid, please ask the user for them`;
  }
}

async function handleResponse(chatEngine, response) {
  try {
    const functionRequest = JSON.parse(response.response);
    if (functionRequest.function === 'favoriteColorTool') {
      // log the function call so that we see when they are called
      console.log('  FUNCTION CALLED WITH: ' + inspect(functionRequest.arguments));

      // call the function requested
      const favColor = getFavoriteColor(functionRequest.arguments.city,
                                        functionRequest.arguments.country);

      // send the response to the chat engine 
      return (handleResponse(chatEngine,
        await chatEngine.chat({message: favColor})));
    } else if (functionRequest.function === 'sendMessage') {
      // LLM sometimes asked to send a message to the user
      return { response: functionRequest.arguments.message };
    } else {
        return (handleResponse(chatEngine,
          await chatEngine.chat({message: 'that function is not available'})));
    }
  }    
  catch {
    // not a function request so just return to the user
    return response;
  }
}


/////////////////////////////
// ASK QUESTIONS
const questions = ['What is my favorite color?', 
                   'My city is Ottawa',
                   'My country is Canada',
                   'I moved to Montreal. What is my favorite color now?',
                   'My city is Montreal and my country is Canada',
                  ];

for (let i = 0; i< questions.length; i++) {
  console.log('QUESTION: ' + questions[i]);
  let response = await chatEngine.chat(getQuery(questions[i]));
  console.log('  RESPONSE:' + (await handleResponse(chatEngine, response)).response);
}                   

import { 
    Ollama, 
    FunctionTool, 
    ReActAgent,
    Settings,
    OpenAI,
    OpenAIAgent,
} from "llamaindex"

import {inspect} from 'util';

Settings.llm = new Ollama({ 
    config: { host: 'http://10.1.1.39:11434' },
    model: 'mistral', // Default value
});

const invalidInfoMessage = { message: 'The city or country was invalid, please ask for the city and country' };
const getFavoriteColor = (info) => {
  console.log(' FUNCTION CALLED WITH: ' + inspect(info));
  if (!info || !(info.city && info.country)) {
    return invalidInfoMessage;
  }

  // return the favorite color based on city and country
  if ((info.city !== 'Ottawa') || (info.country !== 'Canada')) {
    return invalidInfoMessage;
  }

  return {answer: 'black'};
}

const tools = [
    FunctionTool.from(
        getFavoriteColor,
        {
            name: 'favoriteColorTool',
            description: 'return the favorite color for a person based on their city and country',
            parameters: {
                type: 'object',
                properties: {
                    city: {
                        type: 'string',
                        description: 'city'
                    },
                    country: {
                        type: 'string',
                        description: 'country'
                    },
                },
                required: ['city', 'country']
            }
        }
    )
]

function getQuery(query) {
  return { message: `When asked for a favorite color always call favoriteColorTool.
                     A city and country is needed to get the favorite color.
                     When asked for a humans favorite color, if you don't know their city ask for it. 
                     Answer the following question: ${query}` };
}

const agent = new ReActAgent({tools})

const questions = ['What is my favorite color? I live in Ottawa, Canada', 
                   'My city is Ottawa',
                   'My country is Canada' ];

for (let i = 0; i< questions.length; i++) {
  console.log('QUESTION: ' + questions[i]);
  let response = await agent.chat(getQuery(questions[i]));
  console.log('  RESPONSE: ' + response.response.message.content);
}                   

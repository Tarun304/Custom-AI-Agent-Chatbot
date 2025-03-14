from pydantic import BaseModel
from typing import List
from Agents.ai_agent import get_response_from_ai_agent
import uvicorn

# Set up the Schema Validation

class RequestState(BaseModel):
    model_name: str
    system_prompt: str
    messages: List[str]
    allow_search: bool


# AI Agent set up from FrontEnd Request
from fastapi import FastAPI

ALLOWED_MODEL_NAMES=['Gemini', 'Tiny Llama']

app=FastAPI(title='LangGraph AI Agent')

@app.post("/chat")
def chat_endpoint(request: RequestState):
    """
    API Endpoint to interact with the Chatbot using Langraph and search tools.
    It dynamically selects the model specified in the request.

    """

    if request.model_name not in ALLOWED_MODEL_NAMES:
        return { 'error': 'Invalid model name. Kindly select a valid AI model.'}
    
    model_name= request.model_name
    query= request.messages
    allow_search= request.allow_search
    system_prompt=request.system_prompt
    

    # Create AI agent and get response
    response= get_response_from_ai_agent(model_name, query, allow_search, system_prompt)

    return response

if __name__== "__main__":
    uvicorn.run(app, host='127.0.0.1', port=9999)


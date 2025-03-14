import os
from dotenv import load_dotenv 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import  AIMessage



load_dotenv()

# Set up API keys of Google, Hugging Face and Tavily

os.environ['GOOGLE_API_KEY']=os.getenv('GOOGLE_API_KEY')
os.environ['HUGGINGFACEHUB_API_TOKEN']=os.getenv('HUGGINGFACEHUB_API_TOKEN')
os.environ['TAVILY_API_KEY']=os.getenv('TAVILY_API_KEY')

# Set up LLMs and Tools

gemini_llm= ChatGoogleGenerativeAI(model= 'gemini-1.5-pro')

llm= HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)

tinylama_llm= ChatHuggingFace(llm=llm)

search_tool= TavilySearchResults( max_results=2)

# Set up AI Agent with Search tool functionality



def get_response_from_ai_agent(model_name, query, allow_search, system_prompt):
    
    if model_name =='Gemini':
        llm=gemini_llm
    elif model_name =='Tiny Llama':
        llm=tinylama_llm

    tools=[search_tool] if allow_search else []

    # Strenghten the system prompt to enforce domain restrictions
    strict_prompt = (
        f"{system_prompt}\n\n"
        "IMPORTANT: You must ONLY answer questions that are relevant to the given role. "
        "If a question is unrelated to your role, respond with: "
        "'I'm only allowed to answer questions related to my assigned role.'"
    )

    agent=create_react_agent(
        model=llm,
        tools=tools,
        state_modifier= strict_prompt

    )

    state={"messages": query}
    response=agent.invoke(state)
    messages=response.get('messages')
    ai_messages=[message.content for message in messages if isinstance(message, AIMessage)]

    return ai_messages[-1]


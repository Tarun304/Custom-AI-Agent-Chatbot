import streamlit as st
import requests

# Create Steamlit UI

st.set_page_config(page_title='Custom AI Agent', layout='centered')
st.title('AI Chatbot Agent')
st.write('Create and Interact with your Custom AI Agent!')

system_prompt=st.text_area('Define your AI Agent: ', height=70, placeholder='Type your prompt here...')

model_name=st.radio('Select your Large Language Model:', ('Gemini', 'Tiny Llama'))

allow_web_search= st.checkbox('Allow Web Search')

user_query= st.text_area('Enter your query: ', height=70, placeholder='Type your questions here...')

API_URL= 'http://127.0.0.1:9999/chat'

if st.button("Ask Agent!"):
    if user_query.strip():

        payload={

            'model_name': model_name,
            'system_prompt': system_prompt,
            'messages': [user_query],
            'allow_search': allow_web_search
        }

        response= requests.post(API_URL, json=payload)
        if response.status_code == 200:
            response_data= response.json()
            if 'error' in response_data:
                st.error(response_data['error'])
            else:
                st.subheader("Agent Response:")
                st.markdown(f"{response_data}")




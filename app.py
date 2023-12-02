import streamlit as st
from transformers import AutoModelForCausalLM

# Load the model
@st.cache(allow_output_mutation=True)
def load_llm():
    llm = AutoModelForCausalLM.from_pretrained(
        "codellama-7b-instruct.Q2_K.gguf",
        model_type="llama",
        max_new_tokens=1096,
        repetition_penalty=1.13,
        temperature=0.1,
    )
    return llm

llm = load_llm()

# Set title and description
st.title("CodeLlama 7B GGUF")
st.write("This is an enhanced UI for interacting with the CodeLlama 7B model. Please enter your query or select an example to get started.")

# Example queries
examples = [
    "write a code to connect a SQL database and list down all the tables",
    "write a python code for linear regression model using scikit learn",
    "write the code to implement a binary tree implementation in c language",
    "what are the benefits of python programming language?",
    "Create a neural network for vgg16"
]

# Selection box for examples
example_select = st.selectbox("Choose an example query", [""] + examples)

# Text input for user query and chat history
user_query = st.text_input("Your Query", value=example_select)
chat_history = st.text_area("Chat History", height=150)

# Button to get model response
if st.button('Get Response'):
    with st.spinner('Generating response...'):
        # Combining chat history and user query
        full_query = chat_history + "\n" + user_query if chat_history else user_query
        response = llm(full_query)
        st.text_area("Model Response", response, height=150)

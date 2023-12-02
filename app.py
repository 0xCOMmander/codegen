import streamlit as st
from transformers import AutoModelForCausalLM


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


# Initialize the model
llm = load_llm()

st.title("CodeLlama 7B GGUF")

# Text area for user input
user_input = st.text_area("Enter your query:", height=100)
chat_history = st.text_area("Chat History:", height=300)

# Button to generate response
if st.button("Get Response"):
    with st.spinner("Generating response..."):
        response = llm(user_input)
        st.text_area("Model Response", response, height=300)

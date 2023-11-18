import logging
import cohere
import streamlit as st
from datetime import datetime

from completion import Completion
from db import Qdrant, Weaviate

# Configure logger
logging.basicConfig(format="\n%(asctime)s\n%(message)s", level=logging.INFO, force=True)

COHERE_API_KEY = str(st.secrets["COHERE_API_KEY"])

Qdrant.check_collection()

cohere_client_ = cohere.Client(COHERE_API_KEY)


def get_embeddings(text):
    response = cohere_client_.embed(
        texts=[text],
        model='embed-english-v3.0',
        input_type='search_document'
    )
    return response.embeddings[0]


def complete(text, max_tokens, temperature):
    """
    Complete Text.
    """
    if st.session_state.n_requests >= 5:
        st.session_state.text_error = "Too many requests. Please wait a few seconds before completing another Text."
        logging.info(f"Session request limit reached: {st.session_state.n_requests}")
        st.session_state.n_requests = 1
        return

    st.session_state.complete = ""
    st.session_state.text_error = ""
    # st.session_state.visibility = ""
    st.session_state.n_requests = 0

    if not text:
        st.session_state.text_error = "Please enter a text to complete."
        return

    with text_spinner_placeholder:
        with st.spinner("Getting ready your suggestions(can take a few moments)..."):
            # get the embeddings
            embeddings = get_embeddings(text)
            # get similar docs
            lst_contents = Weaviate.get_similar_docs(embeddings)

            similar_contents = ""
            for i in range(len(lst_contents)):
                if i < 1:
                    similar_contents += lst_contents[i]["content"] + "\n"

            completion_ = Completion()
            initial_prompt = f"Remove person names, age, any place name from this content below extract only diseases names and the symptoms the patient is having also extract the steps that has been taken to cure the patients.Return only this values{similar_contents}. \n",  

            extract_infos = completion_.complete(initial_prompt, max_tokens, temperature)
            
            final_prompt = f"Act like a doctors to save humans. The current information of the patients are: {text}. Previously some patients came with this and they are treated with this: \n{extract_infos}. Use those information and your knowledge to give doctor takes necessary steps and give also some follow up questions to ask patients .Do not include any JSOn , HTML body.",  

            completed_text = completion_.complete(final_prompt, 1000, temperature)

            st.session_state.text_error = ""
            st.session_state.n_requests += 1
            st.session_state.complete = (completed_text)
            
            payload = {
                "time": datetime.now(),
                "text": text,
                "initial_prompt":initial_prompt,
                "extract_infos":extract_infos,
                "final_prompt":final_prompt,"completed_text":completed_text
            }
            Qdrant.sent_data(payload)
            

# Configure Streamlit page and state
st.set_page_config(page_title="DoctorsMate🩺", page_icon="")


# Store the initial value of widgets in session state
if "complete" not in st.session_state:
    st.session_state.complete = ""
if "text_error" not in st.session_state:
    st.session_state.text_error = ""
if "n_requests" not in st.session_state:
    st.session_state.n_requests = 0
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"


# Force responsive layout for columns also on mobile
st.write(
    """
    <style>
    [data-testid="column"] {
        width: calc(40% - 1rem);
        flex: 1 1 calc(40% - 1rem);
        min-width: calc(40% - 1rem);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Render Streamlit page
st.title("Welcome to DoctorsMate🩺")

# text
text = st.text_area(label="Please provide patient's info", height = 200, placeholder="Details that is needed for getting a good response.\n - Age and Gender with symptoms and suspected diseases names(if any).\n Example:\nA 25 year old female came with severe joint pain and swelling in both hands for eight months. Morning stiffness more than 1 hour everyday and pain relives gradually by doing regular activities.")

# max tokens
max_tokens = 1024

# temperature
temperature = 0.00

# complete button
st.button(
    label="Generate Suggestions",
    key="generate",
    help="Press to Complete text", 
    type="primary", 
    on_click=complete,
    args=(text, max_tokens, temperature),
    )


text_spinner_placeholder = st.empty()
if st.session_state.text_error:
    st.error(st.session_state.text_error)


if st.session_state.complete:
    st.markdown("""---""")
    st.text_area(label="You can do the followings...", height = 500, value=st.session_state.complete,)

st.markdown("<span style='color: red'>Disclaimer: This AI model may occasionally provide incorrect responses. Always verify with professional guidance and use your judgment when interpreting its advice.</span>", unsafe_allow_html=True)


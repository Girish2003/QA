import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
from langdetect import detect
import speech_recognition as sr
import streamlit as st
import streamlit_authenticator as stauth
import yaml
import gtts as gt
from yaml.loader import SafeLoader
import os
from lingua import Language, LanguageDetectorBuilder
st.set_page_config("Chat PDF") 

languages = [Language.ENGLISH, Language.FRENCH, Language.GERMAN, Language.SPANISH, Language.TAMIL, Language.HINDI]
detector = LanguageDetectorBuilder.from_languages(*languages).build()


with open('./config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)


load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
API_KEY = os.getenv("GOOGLE_API_KEY")


def text_to_speech(text, language):
    tts = gt.gTTS(text, lang=language)
    tts.save("audio.mp3")
    os.system("audio.mp3")


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, you can add some more meaning and do not just tell the response as it is, make sure to provide all the details, if the answer is not in
    provided context just say, "Answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   temperature=0.1)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question, detected_language):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question}
        , return_only_outputs=True)
    # print('response',response)
    st.header("Answer")
    translated = GoogleTranslator(source='en', target=detected_language).translate(response["output_text"])
    print('detected language',detected_language)
    print('translated',translated)
    st.write(translated)
    text_to_speech(translated,detected_language)


def main():
        # login_button = st.button("Already registered ? Login", key="new_login_btn")
        # print(login_button)
        # if login_button is False:
        #     print("Before REgister")
        #     try:
        #             email_of_registered_user, username_of_registered_user, name_of_registered_user = authenticator.register_user(preauthorization=False)
        #             if email_of_registered_user:
        #                 st.success('User registered successfully')
        #                 with open('./config.yaml', 'w') as file:
        #                     yaml.dump(config, file, default_flow_style=False)
        #                 # login_button = True
        #     except Exception as e:
        #             st.error(e)

        # elif login_button is True: 
            print("Before login")
            authenticator.login()
            print("After login")
            if st.session_state["authentication_status"]:
                st.balloons()
                print("session change")
                st.title(f'Welcome *{st.session_state["name"]}*')
                authenticator.logout()
                st.title("Need Assistance ? Let us know more about your queries 💁")
                st.header("Ask your doubts here ⬇️")
                st.markdown("""
                    <style>
                        .stButton>button {
                            margin-top: 28px;
                        }
                    </style>
                    """, unsafe_allow_html=True)
                col1, col2 = st.columns([3, 1])

                with col1:
                    user_question = st.text_input("")

                with col2:
                    button_clicked = st.button("Speak")

                if button_clicked:
                    with st.spinner("Listening..."):
                        r = sr.Recognizer()
                        with sr.Microphone() as source:
                            audio_input = None
                            audio = r.listen(source)
                            try:
                                audio_input = r.recognize_google(audio)
                                st.write("You said:", audio_input)
                                # detected_language = detect(audio_input)
                                language = detector.detect_language_of(audio_input)
                                detected_language = language.iso_code_639_1.name.lower()
                                print('detected language',detected_language)
                                translated = GoogleTranslator(source=detected_language, target='en').translate(audio_input)
                                user_question = translated  
                                user_input(translated, detected_language)
                            except sr.UnknownValueError:
                                st.write("Sorry, could not understand audio.")
                            except sr.RequestError as e:
                                st.write("Error occurred; {0}".format(e))
                        st.success("Listened")
                        

                elif user_question:
                        # detected_language = detect(user_question)
                        language = detector.detect_language_of(user_question)
                        detected_language = language.iso_code_639_1.name.lower()
                        print('language detected',language.iso_code_639_1.name.lower())
                        print('detect 1',detected_language)
                        translated = GoogleTranslator(source=detected_language, target='en').translate(user_question)
                        print('tranlated 1',translated)
                        user_input(translated, detected_language)

                with st.sidebar:
                    st.title("Menu:")
                    pdf_docs = st.file_uploader("Upload your PDF dataset Files and Click on the Submit & Process Button", accept_multiple_files=True)
                    if st.button("Submit & Process"):
                        with st.spinner("Processing..."):
                            raw_text = get_pdf_text(pdf_docs)
                            text_chunks = get_text_chunks(raw_text)
                            get_vector_store(text_chunks)
                            st.success("Done")
    
            elif st.session_state["authentication_status"] is False:
                st.error('Username/password is incorrect')
            elif st.session_state["authentication_status"] is None:
                st.warning('Please enter your username and password')

if __name__ == "__main__":
    main()
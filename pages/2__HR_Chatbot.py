import streamlit as st
def set_page_info():
    st.set_page_config(layout='wide', page_title='HR Coaching', page_icon=':chart_with_upwards_trend:',)
    new_title = '<p style="font-family:sans-serif; color:yellow; font-size: 42px;">HR Coaching</p>'

    st.markdown(new_title, unsafe_allow_html=True)
    st.text("")


set_page_info()

import time
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import FAISS
# prompts
from langchain import PromptTemplate, LLMChain
import textwrap

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_translate():
    model_name = "VietAI/envit5-translation"
    tokenizer = AutoTokenizer.from_pretrained(model_name)  
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer


def translate_en_vi(text:str):
    inputs = [f"en: {text}"]
    outputs = model.generate(tokenizer(inputs, return_tensors="pt", padding=True).input_ids.to('cpu'), max_length=1024) 
    text_output = (tokenizer.batch_decode(outputs, skip_special_tokens=True))
    return text_output, outputs

def translate_vi_en(text:str):
    inputs = [f"vi: {text}"]
    outputs = model.generate(tokenizer(inputs, return_tensors="pt", padding=True).input_ids.to('cpu'), max_length=1024) 
    text_output = (tokenizer.batch_decode(outputs, skip_special_tokens=True))
    return text_output, outputs

class CFG:
    # LLMs
    model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
    temperature = 0.5
    top_p = 0.95
    repetition_penalty = 1.15
    do_sample = True
    max_new_tokens = 1024
    num_return_sequences=1

    # splitting
    split_chunk_size = 800
    split_overlap = 0
    
    # embeddings
    embeddings_model_repo = 'sentence-transformers/all-MiniLM-L6-v2'

    # similar passages
    k = 5
    
    # paths
    Embeddings_path =  './faiss_hp/'

llm = HuggingFaceHub(
    repo_id = CFG.model_name,
    model_kwargs={
        "max_new_tokens": CFG.max_new_tokens,
        "temperature": CFG.temperature,
        "top_p": CFG.top_p,
        "repetition_penalty": CFG.repetition_penalty,
        "do_sample": CFG.do_sample,
        "num_return_sequences": CFG.num_return_sequences        
    },
    huggingfacehub_api_token = st.secrets["hugging_api_key"]
)



# from langchain.embeddings import HuggingFaceInstructEmbeddings
# ### download embeddings model
# embeddings = HuggingFaceInstructEmbeddings(
#     model_name = CFG.embeddings_model_repo,
#     model_kwargs = {"device": "cpu"}
# )

# ### load vector DB embeddings
# vectordb = FAISS.load_local(
#     CFG.Embeddings_path,
#     embeddings
# )

# retriever = vectordb.as_retriever(search_kwargs = {"k": CFG.k, "search_type" : "similarity"})

# we get the context part by embedding retrieval 
# prompt_template_career = """<s>[INST] I want you to act as a career coach. Given the {question}. 
# Suggest 5 portfolio projects and ideas for upgrading the career path and skills for {question}[/INST]"""

# PROMPT_career = PromptTemplate(
#     template = prompt_template_career,
#     input_variables = ["question"]
# )

# # we get the context part by embedding retrieval 
prompt_template_hr= """<s>[INST] I want you to act as a Human Resource coach. Given the {question}. 
Explain the situation of the {question} and provide 5 options to solve and suggest the best solution from the above 5 options with details why need to do it[/INST]"""

PROMPT_hr = PromptTemplate(
    template = prompt_template_hr,
    input_variables = ["question"]
)


# qa_chain = RetrievalQA.from_chain_type(
#     llm = llm,
#     chain_type = "stuff", # map_reduce, map_rerank, stuff, refine
#     retriever = retriever, 
#     chain_type_kwargs = {"prompt": PROMPT_hr},
#     return_source_documents = True,
#     verbose = False
# )

# def wrap_text_preserve_newlines(text, width=700):
#     # Split the input text into lines based on newline characters
#     lines = text.split('\n')

#     # Wrap each line individually
#     wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

#     # Join the wrapped lines back together using newline characters
#     wrapped_text = '\n'.join(wrapped_lines)

#     return wrapped_text


# def process_llm_response(llm_response):
#     ans = wrap_text_preserve_newlines(llm_response['result'])
    
#     sources_used = ' \n \n > '.join(
#         [
#             source.metadata['source'].split('/')[-1][:-4] + ' - page: ' + str(source.metadata['page'])
#             for source in llm_response['source_documents']
#         ]
#     )
    
#     ans = ans + ' \n \n \t Sources: \n >' + sources_used
#     return ans

def llm_hr(question):
    llm_chain = LLMChain(prompt=PROMPT_hr, llm=llm)
    answer = llm_chain.run(question=question)
    return answer.strip()

# def llm_ans(query):
#     start = time.time()
#     llm_response = qa_chain(query)
#     ans = process_llm_response(llm_response)
#     end = time.time()

#     time_elapsed = int(round(end - start, 0))
#     time_elapsed_str = f' \n \n Time elapsed: {time_elapsed} s'
#     return ans.strip() + time_elapsed_str


# question = st.text_input("Asking question:")
# submit = st.button('Generate Answer!')

# st.write("---")

# if submit or question:
#     # col1, col2 = st.columns(2)
#     # with col1:
#     #     st.subheader("Answer from LLM model (Mistral-7B-Instruct-v0.1):")
#     #     with st.spinner(text="This may take a moment..."):
#     #         answer_llm = (llm(f"""<s>[INST] {question} [/INST]""", raw_response=True).strip())
#     #     st.markdown(answer_llm)
#     # with col2:
#     #     st.subheader("Answer from Fine-tunning LLM model:")
#     #     with st.spinner(text="This may take a moment..."):
#     #         answer_tune= llm_ans(question)
#     #     st.markdown(answer_tune)
#     st.subheader("Answer:")
#     with st.spinner(text="This may take a moment..."):
#         answer_tune= llm_ans(question)

if "messages_hr" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages_hr = [
        {"role": "assistant", "content": "Ask me a question about your difficult situation!"}
    ]
if prompt_for_hr := st.chat_input("Your question", key='hr_chat'): # Prompt for user input and save to chat history
    st.session_state.messages_hr.append({"role": "user", "content": prompt_for_hr})

for message in st.session_state.messages_hr: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages_hr[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = llm_hr(prompt_for_hr)
            st.write(response)
            message = {"role": "assistant", "content": response}
            st.session_state.messages_hr.append(message) # Add response to message history


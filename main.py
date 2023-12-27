# import time
# import streamlit as st
# from langchain.chains import RetrievalQA
# from langchain.llms import HuggingFaceHub
# from langchain.vectorstores import FAISS
# # prompts
# from langchain import PromptTemplate, LLMChain
# import textwrap

# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# def load_translate():
#     model_name = "VietAI/envit5-translation"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)  
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#     return model, tokenizer


# def translate(text:str):

#     inputs = [f"en: {text}"]

#     outputs = model.generate(tokenizer(inputs, return_tensors="pt", padding=True).input_ids.to('cpu'), max_length=1024) 

#     text_output = (tokenizer.batch_decode(outputs, skip_special_tokens=True))
#     return text_output, outputs


# class CFG:
#     # LLMs
#     model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
#     temperature = 0.5
#     top_p = 0.95
#     repetition_penalty = 1.15
#     do_sample = True
#     max_new_tokens = 1024
#     num_return_sequences=1

#     # splitting
#     split_chunk_size = 800
#     split_overlap = 0
    
#     # embeddings
#     embeddings_model_repo = 'sentence-transformers/all-MiniLM-L6-v2'

#     # similar passages
#     k = 5
    
#     # paths
#     Embeddings_path =  './faiss_hp/'


# llm = HuggingFaceHub(
#     repo_id = CFG.model_name,
#     model_kwargs={
#         "max_new_tokens": CFG.max_new_tokens,
#         "temperature": CFG.temperature,
#         "top_p": CFG.top_p,
#         "repetition_penalty": CFG.repetition_penalty,
#         "do_sample": CFG.do_sample,
#         "num_return_sequences": CFG.num_return_sequences        
#     },
#     huggingfacehub_api_token = st.secrets["hugging_api_key"]
# )



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

# # we get the context part by embedding retrieval 
# prompt_template_career = """<s>[INST] I want you to act as a career coach. Given the {question}. 
# Suggest 5 portfolio projects and ideas for upgrading the career path and skills for {question}[/INST]"""

# PROMPT_career = PromptTemplate(
#     template = prompt_template_career,
#     input_variables = ["question"]
# )

# # we get the context part by embedding retrieval 
# prompt_template_hr= """<s>[INST] I want you to act as a Human Resource coach. Given the {question}. 
# Explain the situation of the {question} and provide 5 options to solve and suggest the best solution from the above 5 options with details why need to do it[/INST]"""

# PROMPT_hr = PromptTemplate(
#     template = prompt_template_hr,
#     input_variables = ["question"]
# )


# # qa_chain = RetrievalQA.from_chain_type(
# #     llm = llm,
# #     chain_type = "stuff", # map_reduce, map_rerank, stuff, refine
# #     retriever = retriever, 
# #     chain_type_kwargs = {"prompt": PROMPT_hr},
# #     return_source_documents = True,
# #     verbose = False
# # )

# def wrap_text_preserve_newlines(text, width=700):
#     # Split the input text into lines based on newline characters
#     lines = text.split('\n')

#     # Wrap each line individually
#     wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

#     # Join the wrapped lines back together using newline characters
#     wrapped_text = '\n'.join(wrapped_lines)

#     return wrapped_text


# # def process_llm_response(llm_response):
# #     ans = wrap_text_preserve_newlines(llm_response['result'])
    
# #     sources_used = ' \n \n > '.join(
# #         [
# #             source.metadata['source'].split('/')[-1][:-4] + ' - page: ' + str(source.metadata['page'])
# #             for source in llm_response['source_documents']
# #         ]
# #     )
    
# #     ans = ans + ' \n \n \t Sources: \n >' + sources_used
# #     return ans

# def llm_career(question):
#     llm_chain = LLMChain(prompt=PROMPT_career, llm=llm)
#     answer = llm_chain.run(question=question)
#     return answer.strip()

# def llm_hr(question):
#     llm_chain = LLMChain(prompt=PROMPT_hr, llm=llm)
#     answer = llm_chain.run(question=question)
#     return answer.strip()

# # def llm_ans(query):
# #     start = time.time()
# #     llm_response = qa_chain(query)
# #     ans = process_llm_response(llm_response)
# #     end = time.time()

# #     time_elapsed = int(round(end - start, 0))
# #     time_elapsed_str = f' \n \n Time elapsed: {time_elapsed} s'
# #     return ans.strip() + time_elapsed_str


# # question = st.text_input("Asking question:")
# # submit = st.button('Generate Answer!')

# # st.write("---")

# # if submit or question:
# #     # col1, col2 = st.columns(2)
# #     # with col1:
# #     #     st.subheader("Answer from LLM model (Mistral-7B-Instruct-v0.1):")
# #     #     with st.spinner(text="This may take a moment..."):
# #     #         answer_llm = (llm(f"""<s>[INST] {question} [/INST]""", raw_response=True).strip())
# #     #     st.markdown(answer_llm)
# #     # with col2:
# #     #     st.subheader("Answer from Fine-tunning LLM model:")
# #     #     with st.spinner(text="This may take a moment..."):
# #     #         answer_tune= llm_ans(question)
# #     #     st.markdown(answer_tune)
# #     st.subheader("Answer:")
# #     with st.spinner(text="This may take a moment..."):
# #         answer_tune= llm_ans(question)

# col1, col2 = st.columns(2)
# with col1:
#     st.header("HR Coaching")
#     if "messages" not in st.session_state.keys(): # Initialize the chat message history
#         st.session_state.messages = [
#             {"role": "assistant", "content": "Ask me a question about your difficult situation!"}
#         ]
#     if prompt_for_hr := st.chat_input("Your question"): # Prompt for user input and save to chat history
#         st.session_state.messages.append({"role": "user", "content": prompt_for_hr})

#     for message in st.session_state.messages: # Display the prior chat messages
#         with st.chat_message(message["role"]):
#             st.write(message["content"])

#     # If last message is not from assistant, generate a new response
#     if st.session_state.messages[-1]["role"] != "assistant":
#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 response = llm_hr(prompt_for_hr)
#                 st.write(response)
#                 message = {"role": "assistant", "content": response}
#                 st.session_state.messages.append(message) # Add response to message history


# with col2:
#     st.header("Career Coaching")
#     if "messages" not in st.session_state.keys(): # Initialize the chat message history
#         st.session_state.messages = [
#             {"role": "assistant", "content": "Give me your current role with your experience years and your next role target!"}
#         ]

#     if prompt_career_bot := st.chat_input("Your question"): # Prompt for user input and save to chat history
#         st.session_state.messages.append({"role": "user", "content": prompt_career_bot})

#     for message_c in st.session_state.messages: # Display the prior chat messages
#         with st.chat_message(message_c["role"]):
#             st.write(message_c["content"])

#     # If last message is not from assistant, generate a new response
#     if st.session_state.messages[-1]["role"] != "assistant":
#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 response_c = llm_career(prompt_career_bot)
#                 st.write(response_c)
#                 message = {"role": "assistant", "content": response_c}
#                 st.session_state.messages.append(message) # Add response to message history


import streamlit as st
def set_page_info():
    
    st.set_page_config(layout='wide', page_title='Welcome page', page_icon="👩‍👩‍👧‍👧")
    st.image("image/banner.png")
    st.title("WELCOME TO NGHE-NHAN-SU-VN")
    st.markdown(
        """
        Nghề nhân sự Việt Nam được thành lập từ 2019 với khát vọng định nghĩa lại Nghề Nhân sự tại Việt Nam, 
        chia sẻ kiến thức, cập nhật các thông tin mới về việc làm, thị trường lao động 
        và nhất là "Đạo đức Những Người Làm Nghề Nhân sự".

        Ngoài các bài viết, chia sẻ thông tin thì NNS cũng sẽ là đơn vị hỗ trợ, 
        tư vấn các thông tin, giải đáp các thắc mắc của các thành viên tham gia. 
        Đồng thời, trong nỗ lực vươn tới các giải pháp tốt nhất cũng như các 
        nội dung chia sẻ hiệu quả, NNS rất mong nhận được sự quan tâm, 
        đóng góp và chia sẻ "thật" để chúng ta có thể có "kết nối những "giá trị thật" 
        với nhau và cùng nhau.

        Nghề Nhân sự Việt Nam chưa nhận đăng quảng cáo hay giới thiệu dịch vụ cho bất cứ đơn vị, 
        tổ chức nào. Vì vậy, mọi người hạn chế đăng tin spam hoặc seeding có chủ đích. 
        Ngoài ra, NNS cũng chưa duyệt các đăng tuyển dụng nhằm đảm báo tính thống nhất 
        của các nội dung và hiệu quả thông tin truyền tải. Mong các thành viên mới cùng hỗ trợ và hợp tác.

        Để tìm các tài liệu hoặc thông tin liên quan, mọi người có thể sử dụng từ khoá chuyên môn. 

        Trường hợp mong muốn hợp tác hoặc đề nghị cung cấp dịch vụ, mọi người có thể tham khảo các thông tin như sau:

        I - Tư vấn và dịch vụ dành cho Tổ chức

            1- Trung tâm đánh giá năng lực (Assessment Center): Đánh giá năng lực ứng viên, nhân viên và xây dựng lộ trình phát triển nghề nghiệp
            2- Thiết kế, Đào tạo và chuyển giao năng lực Nhân sự: Thu hút tài năng, Học tập và phát triển, Phát triển tổ chức, Quản trị hiệu suất, Chế độ chính sách (Total Rewards), Khung năng lực, Văn hoá doanh nghiệp, Truyền thông nội bộ, Trải nghiệm nhân viên
            3- Xây dựng thương hiệu NTD (EB), Truyền thông Thương hiệu doanh nghiệp
            4- Cung cấp chuyên gia, hợp tác và kết nối các doanh nghiệp, đối tác chiến lược
            5- Nghiên cứu, đo lường sức khoẻ vận hành doanh nghiệp, Đội ngũ Lãnh đạo Quản lý.

        II - Tư vấn và dịch vụ dành cho Cá nhân

            1- Phát triển năng lực, nghề nghiệp và giới thiệu việc làm trọn gói dành cho cá nhân
            2- Xây dựng và phát triển thương hiệu Cá nhân
            3- Xây dựng và phát triển năng lực dành cho Chuyên gia
            4- Đào tạo chuyên môn về Nhân sự, phát triển Căng lực Chuyên sâu
            5- Đào tạo Kỹ năng Quản lý, Lãnh đạo hiệu quả
            6- Hướng dẫn, nhượng quyền Kinh doanh và dịch vụ dành cho cá nhân giúp cải thiện thu nhập và phát triển chuyên môn Nhân sự, Nghề nghiệp.
        
        Trường hợp bạn cần hỗ trợ vấn đề gì, có thể để lại thông tin tại đây: https://bit.ly/nns247
        
        Một lần nữa cảm ơn và chúc các thành viên mới thật nhiều niềm vui và đồng hành cùng NNS trong thời gian dài.
        Trân trọng,


        ********************************
        For more details, please contact: 
        
        Admissions and Consultation Department:
        
        - :telephone_receiver:  0984013214 - 0948013214 

        - :incoming_envelope:   chung@hrd.edu.vn
        
        VIETNAM HUMAN RESOURCES PROFESSIONAL
        - Hotline: :telephone_receiver:  0905 69 89 96 
        - :incoming_envelope:  chung@nghenhansu.vn
    """)
    st.markdown(f"""    
        ---
        # FOLLOW US: :thumbsup:
    """)
    url1 = "https://lnkd.in/gA5xeik"
    st.markdown("       1-Facebook: [https://lnkd.in/gA5xeik](%s)" % url1)

    url2 = "https://lnkd.in/gkxwPSFA"
    st.markdown("       2-LinkedIn: [https://lnkd.in/gkxwPSFA](%s)" % url2)


    url3 = "nghetuyendung.com"
    st.markdown("       3-Recruitment, Training, and Coaching: [nghetuyendung.com](%s)" % url3)

    url4 = "https://zalo.me/nghenhansu"
    st.markdown("       4-HR News on Zalo Official Account:  [https://zalo.me/nghenhansu](%s)" % url4)
             
            
    st.markdown("********************************")           
    st.text("")



set_page_info()



# import time
# import streamlit as st
# from langchain.chains import RetrievalQA
# from langchain.llms import HuggingFaceHub
# from langchain.vectorstores import FAISS
# # prompts
# from langchain import PromptTemplate, LLMChain
# import textwrap

# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# class CFG:
#     # LLMs
#     model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
#     temperature = 0.5
#     top_p = 0.95
#     repetition_penalty = 1.15
#     do_sample = True
#     max_new_tokens = 1024
#     num_return_sequences=1

#     # splitting
#     split_chunk_size = 800
#     split_overlap = 0
    
#     # embeddings
#     embeddings_model_repo = 'sentence-transformers/all-MiniLM-L6-v2'

#     # similar passages
#     k = 5
    
#     # paths
#     Embeddings_path =  './faiss_hp/'

# llm = HuggingFaceHub(
#     repo_id = CFG.model_name,
#     model_kwargs={
#         "max_new_tokens": CFG.max_new_tokens,
#         "temperature": CFG.temperature,
#         "top_p": CFG.top_p,
#         "repetition_penalty": CFG.repetition_penalty,
#         "do_sample": CFG.do_sample,
#         "num_return_sequences": CFG.num_return_sequences        
#     },
#     huggingfacehub_api_token = st.secrets["hugging_api_key"]
# )

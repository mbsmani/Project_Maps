import os
import pickle
import googlemaps
import streamlit as st
from serpapi import GoogleSearch
from urllib.parse import urlsplit, parse_qsl
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from html_template import css,bot_template,user_template,rules

# google suggestions return
def get_place_suggestions(user_input):
    key = os.getenv('GMAPS')
    gmaps = googlemaps.Client(key=key)
    #find autosuggestions
    res = gmaps.places_autocomplete(input_text=user_input,types='establishment',components={'country':['IN']})
    suggestions = {}
    for i in res:
        suggestions[i['description']] = i['place_id']
    return suggestions

# get reviews based on suggestions
def get_reviews(place_id,count=50):
    dirname = 'Reviews'
    if(not os.path.exists(dirname)):
        os.mkdir(dirname)
        print('made dir reviews')

    if(os.path.exists(dirname+'/{}.pkl'.format(place_id))):
        with open(dirname+'/{}.pkl'.format(place_id),'rb') as f:
            reviews = pickle.load(f)
        f.close()

        return reviews
    
    key = os.getenv('SERPAPI_2')
    
    params = {
    "api_key": key,                     # your api key
    "engine": "google_maps_reviews",    # serpapi search engine
    "hl": "en",                         # language of the search
    "place_id": place_id,                # google place id
    "no_cache":True
    }

    search = GoogleSearch(params)
    reviews = []
    page_num = 0

    while True:
        page_num += 1
        results = search.get_dictionary()
        
        if not "error" in results:
            for result in results.get("reviews", []): # return an empty list [] if no reviews from the place
                reviews.append({
                    "page": page_num,
                    "name": result.get("user").get("name"),
                    "link": result.get("user").get("link"),
                    "thumbnail": result.get("user").get("thumbnail"),
                    "rating": result.get("rating"),
                    "date": result.get("date"),
                    "text": result.get("snippet"),
                    "images": result.get("images"),
                    "local_guide": result.get("user").get("local_guide"),
                })
        else:
            st.error(results["error"])
            break

        if(len(reviews) >= count):
            break

        if results.get("serpapi_pagination").get("next") and results.get("serpapi_pagination").get("next_page_token"):
            # split URL in parts as a dict and update search "params" variable to a new page that will be passed to GoogleSearch()
            search.params_dict.update(dict(parse_qsl(urlsplit(results["serpapi_pagination"]["next"]).query)))
        else:
            break

    with open(dirname+'/{}.pkl'.format(place_id),'wb') as f:
        pickle.dump(reviews,f)
    f.close()
    
    return reviews

# vectorize results
def get_vector_store():
    dirname = 'Embedded_Docs'
    emb = HuggingFaceEmbeddings(model_name='thenlper/gte-large')
    place_id = st.session_state.suggestions[st.session_state.choice]
    
    if(not os.path.exists(dirname)):
        os.mkdir(dirname)
        print('made dir')

    if(os.path.exists(dirname+'/{}.pkl'.format(place_id))):
        print('from cache')
        vect_str = FAISS.load_local(dirname,embeddings=emb,index_name=place_id)
        return vect_str
    
    # get reviews for the place
    st.session_state.reviews = get_reviews(place_id=place_id)            
    chunks = [i['text'] for i in st.session_state.reviews if(i['text'] is not None)]
    vect_str = FAISS.from_texts(texts=chunks,embedding=emb)
    vect_str.save_local(folder_path='Embedded_Docs',index_name=place_id)
    return vect_str

# get conversation chain
def get_conv_chain(vect_store,temp=0.5):
    prompt_template = """Use the following pieces of context to answer the question at the end within 100 words. If you don't know the answer, just say that you don't know, don't try to make up an answer.

                    {context}

                    Question: {question}
                    Answer:"""
    finalPrompt = PromptTemplate(
                                template=prompt_template, 
                                input_variables=['context', 'question'])
    
    mem = ConversationBufferMemory(memory_key='chat_history',
                                   output_key='answer',
                                   return_messages=True)
    
    my_llm = ChatOpenAI(temperature=temp) 

    conv = ConversationalRetrievalChain.from_llm(llm=my_llm,
                                                 memory=mem,
                                                 retriever=vect_store.as_retriever(),
                                                 combine_docs_chain_kwargs={'prompt':finalPrompt},
                                                 rephrase_question=True,
                                                 return_source_documents=False,
                                                 return_generated_question=False)
    return conv

# process questions and display results
def handle_question(query):
    resp = st.session_state.conv({'question':query})
    st.session_state.chat_hist = resp['chat_history']
    for i,msg in enumerate(st.session_state.chat_hist):
        if(i%2 == 0):
            st.write(user_template.replace('{{MSG}}',msg.content),unsafe_allow_html=True)
        else:
            st.write(bot_template.replace('{{MSG}}',msg.content),unsafe_allow_html=True)

# main app page
def main():
    st.set_page_config(page_title='YAMI')
    st.write(css,unsafe_allow_html=True)
    if 'choice' not in st.session_state:
        st.session_state.choice = ''
    if 'suggestions' not in st.session_state:
        st.session_state.suggestions = {}
    if 'reviews' not in st.session_state:
        st.session_state.reviews = []
    if 'vect_store' not in st.session_state:
        st.session_state.vect_store = None
    if 'conv' not in st.session_state:
        st.session_state.conv = None
    if 'chat_hist' not in st.session_state:
        st.session_state.chat_hist = None

    st.header('Y:red[A]TRA M:red[I]TRA (Y:red[a]M:red[i]) - _Your Destination Companion!_')
    # get questions on places
    query = st.text_input(label='Enter your questions:',placeholder='Does this place have road access?')

    if query:
        handle_question(query)

    with st.sidebar:
        st.header('How to Use:')
        st.text(rules)
        st.header('Place Search')
        user_input = st.text_input(label='Place Name:',placeholder='Taj Mahal')
        if(st.button('GO',key='b1')):
            st.session_state.chat_hist = None
            with st.spinner('Getting places...'):
                st.session_state.suggestions = get_place_suggestions(user_input)
        place = st.selectbox(label='Choose the place:',options=st.session_state.suggestions.keys())
        if(st.button('GO',key='b2')):
            spinner_str = place.split(',')[0]
            with st.spinner('Exploring {}, please wait!'.format(spinner_str)):
                st.session_state.choice = place
                # vectorize reviews
                st.session_state.vect_store = get_vector_store()
                # get converstion
                st.session_state.conv = get_conv_chain(st.session_state.vect_store)

if __name__ == '__main__':
    main()
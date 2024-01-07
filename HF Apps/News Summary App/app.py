import gradio as gr
from wordcloud import WordCloud
import pandas as pd
import requests
import json
import hopsworks
import matplotlib.pyplot as plt
import os 
import time

MODEL = "gpt-3.5-turbo"
API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


project = hopsworks.login(project="ServerlessIntroIris")
fs = project.get_feature_store()
dataset1 = fs.get_feature_group(name="daily_topic_info").read()
df = dataset1
dataset2 = fs.get_feature_group(name="daily_document_info").read()
df2 = dataset2
topics = df['topic'].unique()

def gpt_predict(inputs, request:gr.Request=gr.State([]), top_p = 1, temperature = 1, chat_counter = 0,history =[]):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": f"{inputs}"}],
        "temperature" : 1.0,
        "top_p":1.0,
        "n" : 1,
        "stream": True,
        "presence_penalty":0,
        "frequency_penalty":0,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }

    # print(f"chat_counter - {chat_counter}")
    if chat_counter != 0 :
        messages = []
        for i, data in enumerate(history):
            if i % 2 == 0:
                role = 'user'
            else:
                role = 'assistant'
            message = {}
            message["role"] = role
            message["content"] = data
            messages.append(message)
        
        message = conversation_history
        messages.append(message)
        payload = {
            "model": MODEL,
            "messages": messages,
            "temperature" : temperature,
            "top_p": top_p,
            "n" : 1,
            "stream": True,
            "presence_penalty":0,
            "frequency_penalty":0,
        }

    chat_counter += 1

    history.append(inputs)
    token_counter = 0 
    partial_words = "" 
    counter = 0

    try:
        # make a POST request to the API endpoint using the requests.post method, passing in stream=True
        response = requests.post(API_URL, headers=headers, json=payload, stream=True)
        response_code = f"{response}"
        
        if response_code.strip() != "<Response [200]>":
            #print(f"response code - {response}")
            raise Exception(f"Sorry, hitting rate limit. Please try again later. {response}")
        out = []
        for chunk in response.iter_lines():
            #Skipping first chunk
            if counter == 0:
                counter += 1
                continue
                #counter+=1
            # check whether each line is non-empty
            if chunk.decode() :
                chunk = chunk.decode()
                # decode each line as response data is in bytes
                if len(chunk) > 12 and "content" in json.loads(chunk[6:])['choices'][0]['delta']:
                    partial_words = partial_words + json.loads(chunk[6:])['choices'][0]["delta"]["content"]
                    if token_counter == 0:
                        history.append(" " + partial_words)
                    else:
                        history[-1] = partial_words
                    token_counter += 1
    except Exception as e:
        print (f'error found: {e}')
    return partial_words

readable_topics_dic = dict()
input = "I have lists of multiple words : " 
mrk = [] #most representative keyword (used if chatgpt doesn't work)
for t in topics.tolist():
    if t != -1:
        selected_data = df[df['topic'] == t]
        keywords = selected_data['keywords'][selected_data.index[0]]
        freq = selected_data["scores"][selected_data.index[0]]
        keyword_freq_pairs = zip(keywords, freq)
        most_frequent_keyword = max(keyword_freq_pairs, key=lambda x: x[1])
        print(most_frequent_keyword[0].capitalize())
        mrk.append(most_frequent_keyword[0].capitalize())
        input += ", [" + ", ".join(keywords) + "]"


input += "  I want you to give me only one precise word that best describes the theme of this list. If I give you multiple lists, I want you to give me one word with a maj in front for each of those lists, and separate them by // (your answer should contain only one word for each, if I give you 100 lists, You give me 100 words)"
new_topics = "".join(gpt_predict(input)) 
nt = new_topics.split("//")

#in case chatgpt overloaded
i = 0
if len(nt) < len(topics.tolist()):
    nt = [f"Topic {o+1}: {mrk[o]}" for o in range(len(topics.tolist())-1)]
for t in topics.tolist():
    if t != -1:
        readable_topics_dic[nt[i]] = t
        if i < len(nt)-1:
            i += 1



def display_topics(topic):
    topic = readable_topics_dic[topic]
    # Filter DataFrame based on the selected topic
    selected_data = df[df['topic'] == topic]
    selected_data2 = df2[df2['topic'] == topic]
    selected_data2 = selected_data2.sort_values(by='probability')
    # Display relevant articles
    articles = selected_data2['title'] 
    links = selected_data2['link']
    nb_art = min(4, len(links))
    articles_ret = """## Most relevant articles  
      
    """
    for i in range(nb_art):
        ind = articles.index[i]
        articles_ret += f""" * [{articles[ind]}]({links[ind]})  
    """
    # Generate word cloud for keywords
    keywords = selected_data['keywords'][selected_data.index[0]]
    freq = selected_data["scores"][selected_data.index[0]]
    keywords_wordcloud = dict()
    for i, elem in enumerate(keywords):
        keywords_wordcloud[elem] = freq[i]
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(keywords_wordcloud)
    fig, ax = plt.subplots()
    plt.axis("off")
    ax =plt.imshow(wordcloud, interpolation='bilinear')

    return articles_ret ,  fig

# Define Gradio interface
iface = gr.Interface(
    fn=display_topics,
    inputs=gr.Dropdown(nt, label="Topic"),
    outputs=[gr.Markdown(label="Most relevant articles"),gr.Plot(label="Main Keywords")],
    live=True,
    examples=[]
)
 
# Launch the app
iface.launch()

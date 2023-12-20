
import json
from newspaper import Article

results = []
for i in range(3):
    for j in range(8):
        with open("object"+str(i) + str(j)+".json", 'r') as file:
        # Use json.load() to deserialize and load the object from the file
            response = json.load(file)
            result = response['results']
            for article in response['results']:
                """ try:
                    a = Article(article['link'], language='en')  
                    a.download()
                    a.parse()
                    text = a.text
                except:
                    text = ""

                results.append({'id':article['article_id'], 
                                'title':article['title'],
                                'link':article['link'],
                                'content':article['content'] if len(article['content']) >= len(text) else text}) """
                
                results.append({'id':article['article_id'], 
                                'title':article['title'],
                                'link':article['link'],
                                'content':article['content']})
            
            
        
        pass
#print(results[0]['content'])
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
docs = []
for result in results:
    docs.append(result['content'])
#article = results[0]['content']
    
import time
start_time = time.time()
summary  = summarizer(docs, max_length=225, min_length=150, do_sample=False)
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")
pass
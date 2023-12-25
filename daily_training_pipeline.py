from bertopic import BERTopic
import pandas as pd
import numpy as np
import hopsworks
import joblib
import modal
from modal import Stub, Volume, Image
from datetime import date, timedelta, datetime



def load_hopsworks_model(model_registry, name:str, version:int = 1):
    model = model_registry.get_model(name, version = version)
    model_dir = model.download()
    model = joblib.load(model_dir + "/"+name+".pkl")
    return model

def save_dataframe(df, fs, name:str, version:int = 1, description:str = "", overwrite:bool = True):
  columns = df.columns.tolist()

  fg = fs.get_or_create_feature_group(
      name=name,
      version=version,
      primary_key=columns,
      description=description)
  fg.insert(df, overwrite = overwrite)



def f():
    project = hopsworks.login()
    fs = project.get_feature_store()

    news_fg = fs.get_feature_group(name="news", version=1)
    query = news_fg.select_all()

    feature_view = fs.get_or_create_feature_view(name="news_view",
                                    version=1,
                                    description="Read from news dataset",
                                    query=query)
    
    
    #Get the data uploaded today
    #Only seems to work with tomorrow as end time
    today = datetime.now().date()
    start_date = today
    tomorrow = today + timedelta(days=1)

    df = feature_view.get_batch_data(
            start_time=start_date,
            end_time=tomorrow
        )

    mr = project.get_model_registry()
        

    embedding_model = load_hopsworks_model(mr, "news_embedding", version = 1)
    umap_model = load_hopsworks_model(mr, "news_umap", version = 1)
    hdbscan_model = load_hopsworks_model(mr, "news_hbdscan", version = 1)
    vectorizer_model = load_hopsworks_model(mr, "news_vectorizer", version = 1)
    representation_model = load_hopsworks_model(mr, "news_representation", version = 1)

    topic_model = BERTopic(
    # Pipeline models
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    # Hyperparameters
    top_n_words=10,
    verbose=True
    )

    documents = df['content'].values
    embeddings = np.array(df['embedding'].values.tolist())


    topics, probs = topic_model.fit_transform(documents, embeddings)
    #Find proper keywords for basic topics
    topic_model.update_topics(documents, representation_model=representation_model)

    info = topic_model.get_topic_info()
    topic_scores = topic_model.get_topics()

    titles = df['title'].values
    links = df['link'].values

    titles = titles[documents != None]
    links = links[documents != None]
    document_data = {"topic":topics, 
                "probability":probs, 
                "title":titles, 
                "link":links}
    document_dataframe = pd.DataFrame(document_data)


    topic_dataframe = info
    keywords = [None]*len(topic_scores)
    scores = [None]*len(topic_scores)
    for topic, values in topic_scores.items():
        keyword = [value[0] for value in values]
        score = [value[1] for value in values]

        keywords[topic + 1] = keyword
        scores[topic + 1] = score
    topic_dataframe["keywords"] = keywords
    topic_dataframe["scores"] = scores

    save_dataframe(document_dataframe,
                fs,
                name = "daily_document_info",
                version = 1,
                description="info about today's news documents")

    save_dataframe(topic_dataframe,
                fs,
                name = "daily_topic_info",
                version = 1,
                description="topic summary of today's news")
    

   
stub = Stub(name = "daily_training")

image = Image.debian_slim(python_version="3.10").pip_install_from_requirements(
    requirements_txt="training_requirements.txt",
    extra_index_url="https://download.pytorch.org/whl/cu118"
) 

@stub.function(image=image, 
               gpu="t4",
            schedule=modal.Period(days = 1),
            secrets=[modal.Secret.from_name("HOPSWORKS_API_KEY"),
                                modal.Secret.from_name("NEWSDATA")])
def g():
    f()

    
if __name__ == "__main__":
    with stub.run():
        g.remote()
   




import hopsworks
from hsfs.feature_view import FeatureView
import modal
from modal import Stub, Image

"""
Removes the news articles collected over the last week to prevent exceeding the storage of the feature store
and to keep the stored news relevant
"""

def delete_feature_group(fs, name:str, version:int):
    try:
        fg = fs.get_feature_group(
            name=name,
            version=version,
        )

        fg.delete()
    except:
        print("unable to delete feature group " + name)

def delete_feature_view(fs, name:str, version:int):
    try:
        FeatureView.clean(fs.id, name, version)
        fv = fs.get_feature_view(
            name=name,
            version=version,
        )
        
        fv.delete()
    except Exception as e:
        print("unable to delete feature view " + name)
        print(f"An error occurred: {e}")

def reset_news_project():
    project = hopsworks.login()
    fs = project.get_feature_store()
    dataset_api = project.get_dataset_api()

    dataset_api.remove("Resources/FinalProject/NEWSDATAFeed.pkl")

    delete_feature_group(fs, 
                         name="daily_document_info", 
                         version=1)
    
    delete_feature_group(fs, 
                         name="daily_topic_info", 
                         version=1)
    
    delete_feature_group(fs, 
                         name="news", 
                         version=1)
    
    delete_feature_view(fs, 
                         name="daily_document_info_view", 
                         version=1)
    
    delete_feature_view(fs, 
                         name="daily_topic_info_view", 
                         version=1)
    
    delete_feature_view(fs, 
                         name="news_view", 
                         version=1)



stub = Stub(name = "weekly_reset")
image = Image.debian_slim(python_version="3.10").pip_install(["hopsworks==3.4.3",
                                        "hsfs==3.4.5"]) 

@stub.function(image=image, schedule=modal.Period(days = 7), secrets=[modal.Secret.from_name("HOPSWORKS_API_KEY")])
def g():
    reset_news_project()

    
if __name__ == "__main__":
    with stub.run():
        g.remote()

    
    
    

    





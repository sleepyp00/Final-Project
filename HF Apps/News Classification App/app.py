import gradio as gr
import hopsworks
import torch
import joblib

try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except:
    device = "cpu"

project = hopsworks.login()
fs = project.get_feature_store()

mr = project.get_model_registry()
classifier = mr.get_model("finetuned_classifier", version = 1)
model_dir = classifier.download()
classifier = joblib.load(model_dir + "/finetuned_classifier.pkl")

embedding_model = mr.get_model("news_embedding", version = 1)
model_dir = embedding_model.download()
embedding_model = joblib.load(model_dir + "/news_embedding.pkl")

index_to_category = {
    0:"Polititcs",
    1:"Science",
    2:"Entertainment",
    3:"Sports",
    4:"Business"
}

description = """
This app will provide classifications for text from a news article. 
The input is currently truncated at around 400 words so make sure to include the most important part of the article.
"""



def predict(text):
    embedding = embedding_model.encode([text], device = device)
    with torch.no_grad():
        embedding = torch.tensor(embedding, device=device, dtype=torch.float32)
        probs = classifier.probabilities(embedding).cpu().numpy()
    return {index_to_category[i]: float(conf) for i, conf in enumerate(probs[0])}


gr.Interface(
    predict,
    inputs=gr.Textbox(label="Article"),
    outputs="label",
    theme="huggingface",
    description=description,
).launch()
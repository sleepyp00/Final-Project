import numpy as np 
import hopsworks
import torch
from torch.utils.data import TensorDataset, DataLoader
from utils import train_model, accuracy, ECE, replace_model
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from training_setup import Training_Setup
import modal
from modal import Stub, Image
from finetune_net import Network




def f():
    N_CLASSES = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    project = hopsworks.login()
    fs = project.get_feature_store()

    news_fg = fs.get_feature_group(name="news", version=1)
    query = news_fg.select_all()

    feature_view = fs.get_or_create_feature_view(name="news_view",
                                    version=1,
                                    description="Read from news dataset",
                                    query=query)

    try:
        feature_view.delete()
    except:
        print("No old view to delete")

    feature_view = fs.get_or_create_feature_view(name="news_view",
                                    version=1,
                                    description="Read from news dataset",
                                    query=query)

    data, _ = feature_view.training_data()

    #Just in case, should not occur
    data = data.drop_duplicates(subset=['title'])

    features = np.array(data["embedding"].values.tolist())
    targets = data["category"].values

    features = features[targets != -1,:]
    targets = targets[targets != -1]

    unique_values, counts = np.unique(targets, return_counts=True)

    weights = counts.sum()/(N_CLASSES*counts)
    weights = torch.tensor(weights, device=device, dtype=torch.float32)

    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.1, stratify=targets)

    X_train = torch.tensor(X_train, device=device, dtype=torch.float32)
    X_test = torch.tensor(X_test, device=device, dtype=torch.float32)

    y_train = torch.tensor(y_train, device=device)
    y_test = torch.tensor(y_test, device=device)

    fs = project.get_feature_store()

    mr = project.get_model_registry()
    model = mr.get_model("base_classifier", version = 1)
    model_dir = model.download()
    model = joblib.load(model_dir + "/base_classifier.pkl")

    model.to(device=device)
    model.train()

    training_setup = Training_Setup(
            lr = 1e-3,
            weight_decay=1e-4,
            gamma=0.1,
            milestones=[25, 100, 150],
            weights=weights
    )
    epochs = 200
    batch_size = 64

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    dataloaders = {
        "train":train_loader,
        "validation":test_loader
    }

    train_model(model=model, 
                epochs=epochs, 
                training_setup=training_setup,
                dataloaders=dataloaders)


    model.eval()
    with torch.no_grad():
        acc = accuracy(model, test_loader)
        ece = ECE(model, test_loader)
        predictions = model.probabilities(X_test.to(device))
        predictions = predictions.cpu()
        y_test = y_test.cpu()

    predicted_labels = np.argmax(predictions.numpy(), axis=1)
    true_labels = y_test.numpy()

    # Create the confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    print("accuracy:", acc)
    print("ece:", ece)

    # Print or use the confusion matrix as needed
    print("Confusion Matrix:")
    print(conf_matrix)

    replace_model(mr,
                model.cpu(),
                name="finetuned_classifier",
                version=1,
                description="This model is finetuned recent news data and is updated continously",
                metrics= {"accuracy":acc})



stub = Stub(name = "daily_classifier_training")

image = Image.debian_slim(python_version="3.10").pip_install_from_requirements(
    requirements_txt="training_requirements.txt",
    extra_index_url="https://download.pytorch.org/whl/cu118"
) 

@stub.function(image=image, 
               gpu="t4",
            schedule=modal.Period(days = 1),
            secrets=[modal.Secret.from_name("HOPSWORKS_API_KEY")])
def g():
    f()

    
if __name__ == "__main__":
    with stub.run():
        g.remote()
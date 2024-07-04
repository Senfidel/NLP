import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
fig = px.box(range(10))
import pandas as pd
import bert
import bertopic
import wos
import glob
import wosfile
import bertviz
import spacy
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import openai
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, OpenAI, PartOfSpeech
import joblib
import numpy as np
import gdown

#import bertopic_base_chinese
df=pd.read_csv('df_avec_topics_pays_villes(1).csv')

text=df[ 'Résumé_translated']
# Chemin de chargement


# Télécharger le modèle depuis Google Drive
@st.cache_resource
def download_model(url, output):
    gdown.download(url, output, quiet=False)
    return output

# URL de téléchargement direct de Google Drive
url = 'https://drive.google.com/uc?id=1RZio93kAznqrl9BnxbWgh7NVJNEztFuF' 

output = 'bertopic_model.pkl'
# Chemin du fichier modèle téléchargé
model_path = download_model(url, output)
topic_model = joblib.load(model_path)



# Charger les embeddings
embeddings = np.load(f"{embeddings.npy", allow_pickle=True})

#topic_model = BERTopic.load(r"C:\Users\etien\OneDrive\Documents\Master 2\D2SN\Machine learning\Mémoire\Data\Bert_Model\bertopic_model.pkl", embedding_model="sentence-transformers/all-MiniLM-L6-v2")
#topic_distr, _ = topic_model.approximate_distribution(text, window=8, stride=4)
#fig1=topic_model.visualize_distribution(topic_distr[100], custom_labels=True)
#t.plotly_chart(fig1)
# Visualize topics with custom labels
#fig2=topic_model.visualize_topics(custom_labels=True)
#st.plotly_chart(fig2)

#fig3=topic_model.visualize_hierarchy(custom_labels=True)
#st.plotly_chart(fig3)

#embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
#embeddings = embedding_model.encode(text, show_progress_bar=True)
#reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
#fig4=topic_model.visualize_documents(text, reduced_embeddings=reduced_embeddings, custom_labels=True)
#st.plotly_chart(fig4)
#TensorRT-LLM/Devoir/Preesentation.py



# Streamlit visualisation
st.title("BERTopic Model Visualisations")

# Visualize topics with custom labels
st.plotly_chart(topic_model.visualize_topics(custom_labels=True))

# Visualisation des topics avec des barres
st.plotly_chart(topic_model.visualize_barchart(top_n_topics=10))

# Visualisation des topics au fil du temps
topics_over_time = topic_model.topics_over_time(df[ 'Résumé_translated'], df['Année de Publication'])
st.plotly_chart(topic_model.visualize_topics_over_time(topics_over_time))

# Visualisation des heatmaps des termes
st.plotly_chart(topic_model.visualize_heatmap())

# Visualisation des hiérarchies de sujets
st.plotly_chart(topic_model.visualize_hierarchy())

# Reduce dimensionality of embeddings, this step is optional but much faster to perform iteratively:
reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
st.plotly_chart(topic_model.visualize_documents(text, reduced_embeddings=reduced_embeddings, custom_labels=True))

# Visualisation des documents en fonction des topics

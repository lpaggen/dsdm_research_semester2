from transformers import AutoModelForMaskedLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer, models
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
import matplotlib.pyplot as plt
import umap
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_texts_from_dir(directory):
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            path = os.path.join(directory, filename)
            with open(path, "r", encoding="utf-8") as f:
                texts.append(f.read())
    return texts

corpus = load_texts_from_dir("F:/CodingProjects/dsdm_research_semester2/all_papers")
print(f"loaded {len(corpus)} documents.")

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = Dataset.from_dict({"text": corpus})

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15 # change to desired value, 0.15 is standard for BERT fine-tuning
)

model = AutoModelForMaskedLM.from_pretrained(model_name)

training_args = TrainingArguments(
    output_dir="./bert-mlm-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
    logging_steps=250,
    eval_strategy="no",
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=500,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator
)

trainer.train()

trainer.save_model("./bert-mlm-finetuned")
tokenizer.save_pretrained("./bert-mlm-finetuned")

words = ["quantum", "Hilbert", "topological", "majorana"]

base_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
base_model = BertModel.from_pretrained("bert-base-uncased")

custom_model_path = "F:/CodingProjects/dsdm_research_semester2/python/bert-mlm-finetuned"  # contains config.json, pytorch_model.bin, vocab.txt or tokenizer
custom_tokenizer = BertTokenizer.from_pretrained(custom_model_path)
custom_model = BertModel.from_pretrained(custom_model_path)

def get_embedding(model, tokenizer, word):
    inputs = tokenizer(word, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :] # CLS token embedding
    return cls_embedding.squeeze().numpy()

def compare_similarities(word1, word2):
    emb1_base = get_embedding(base_model, base_tokenizer, word1) # here base bert is used
    emb2_base = get_embedding(base_model, base_tokenizer, word2)
    sim_base = cosine_similarity([emb1_base], [emb2_base])[0][0]

    emb1_ft = get_embedding(custom_model, custom_tokenizer, word1) # here fine-tuned bert is used
    emb2_ft = get_embedding(custom_model, custom_tokenizer, word2)
    sim_ft = cosine_similarity([emb1_ft], [emb2_ft])[0][0]

    print(f"\nwords: {word1} ↔ {word2}")
    print(f"base BERT similarity:       {sim_base:.4f}")
    print(f"fine-tuned BERT similarity: {sim_ft:.4f}")

compare_similarities("quantum", "Hilbert")
compare_similarities("topological", "majorana")
compare_similarities("quantum", "topological")

model_name = "allenai/scibert_scivocab_uncased" # here SCIBERT
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

block_size = 512 # or 512 if you want max capacity

# this function groups the tokenized texts into blocks of a specified size, for SBERT 
def group_texts(examples):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated["input_ids"])
    total_length = (total_length // block_size) * block_size

    result = {
        k: [concatenated[k][i:i+block_size] for i in range(0, total_length, block_size)]
        for k in concatenated.keys()
    }
    return result

lm_dataset = tokenized_datasets.map(group_texts, batched=True)

training_args = TrainingArguments(
    output_dir="./scibert-mlm",
    overwrite_output_dir=True,
    num_train_epochs=30,
    fp16=True,
    per_device_train_batch_size=16,
    save_steps=10_000,
    save_total_limit=2,
    logging_steps=250,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=500,
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

trainer.save_model("./scibert-mlm")
tokenizer.save_pretrained("./scibert-mlm")

words = ["quantum", "Hilbert", "topological", "majorana"]

base_bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
base_bert_model = BertModel.from_pretrained("bert-base-uncased")

ft_bert_path = "./bert-mlm-finetuned" 
ft_bert_tokenizer = BertTokenizer.from_pretrained(ft_bert_path)
ft_bert_model = BertModel.from_pretrained(ft_bert_path)

base_scibert_tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
base_scibert_model = BertModel.from_pretrained("allenai/scibert_scivocab_uncased")

ft_scibert_path = "./scibert-mlm" 
ft_scibert_tokenizer = BertTokenizer.from_pretrained(ft_scibert_path)
ft_scibert_model = BertModel.from_pretrained(ft_scibert_path)

def get_embedding(model, tokenizer, word):
    inputs = tokenizer(word, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :] # CLS
    return cls_embedding.squeeze().numpy()

def compare_all_models(word1, word2):
    emb_base_bert_1 = get_embedding(base_bert_model, base_bert_tokenizer, word1)
    emb_base_bert_2 = get_embedding(base_bert_model, base_bert_tokenizer, word2)
    
    emb_ft_bert_1 = get_embedding(ft_bert_model, ft_bert_tokenizer, word1)
    emb_ft_bert_2 = get_embedding(ft_bert_model, ft_bert_tokenizer, word2)
    
    emb_base_scibert_1 = get_embedding(base_scibert_model, base_scibert_tokenizer, word1)
    emb_base_scibert_2 = get_embedding(base_scibert_model, base_scibert_tokenizer, word2)
    
    emb_ft_scibert_1 = get_embedding(ft_scibert_model, ft_scibert_tokenizer, word1)
    emb_ft_scibert_2 = get_embedding(ft_scibert_model, ft_scibert_tokenizer, word2)

    sim_base_bert = cosine_similarity([emb_base_bert_1], [emb_base_bert_2])[0][0]
    sim_ft_bert = cosine_similarity([emb_ft_bert_1], [emb_ft_bert_2])[0][0]
    sim_base_scibert = cosine_similarity([emb_base_scibert_1], [emb_base_scibert_2])[0][0]
    sim_ft_scibert = cosine_similarity([emb_ft_scibert_1], [emb_ft_scibert_2])[0][0]

    print(f"\nwords: {word1} ↔ {word2}")
    print(f"base BERT similarity:        {sim_base_bert:.4f}")
    print(f"fine-tuned BERT similarity:  {sim_ft_bert:.4f}")
    print(f"base SciBERT similarity:     {sim_base_scibert:.4f}")
    print(f"fine-tuned SciBERT similarity:{sim_ft_scibert:.4f}")

compare_all_models("quantum", "Hilbert")
compare_all_models("topological", "majorana")
compare_all_models("quantum", "topological")

model_path = "F:/CodingProjects/dsdm_research_semester2/python/scibert-mlm"

hf_model = AutoModel.from_pretrained(model_path)
hf_tokenizer = AutoTokenizer.from_pretrained(model_path)

word_embedding_model = models.Transformer(model_path, max_seq_length=512)
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True
)

sentence_model = SentenceTransformer(modules=[word_embedding_model, pooling_model]) # use scibert in the SBERT

data = []

for paper in os.listdir("F:/CodingProjects/dsdm_research_semester2/new_classified/2024"):
    if paper.endswith(".txt"):
        with open(os.path.join("F:/CodingProjects/dsdm_research_semester2/new_classified/2024", paper), "r") as f:
            text = f.read()
        data.append(text)
        
data = [paper for paper in data if paper and paper.strip()]
embedding_model = SentenceTransformer(model_path)
embeddings = embedding_model.encode(data, batch_size=16, show_progress_bar=True)

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def remove_stopwords(text): # the function to remove stopwords from the text
    return ' '.join([word for word in text.split() if word.lower() not in stop_words])

cleaned_docs = [remove_stopwords(doc) for doc in data]
embedding_model = SentenceTransformer(model_path)
embeddings = embedding_model.encode(cleaned_docs, batch_size=16, show_progress_bar=True)
vectorizer_model = CountVectorizer(stop_words='english')
topic_model = BERTopic(vectorizer_model=vectorizer_model)
topics, probs = topic_model.fit_transform(cleaned_docs, embeddings)

print(topic_model.get_topic_info()) # gives info about the topics

umap_reducer = umap.UMAP(n_components=2, init='random', random_state=42) # here we use UMAP to reduce the dimensionality of the embeddings, can try others 
embeddings_2d = umap_reducer.fit_transform(embeddings)

plt.figure(figsize=(12,8))
scatter = plt.scatter(
    embeddings_2d[:,0], embeddings_2d[:,1], 
    c=topics, cmap='tab20', s=15, alpha=0.7
)
plt.colorbar(scatter, label='Topic')
plt.title("Topic Clusters Visualization (2D UMAP)")
plt.xlabel("UMAP dimension 1")
plt.ylabel("UMAP dimension 2")
plt.show()

topics, _ = topic_model.fit_transform(cleaned_docs, embeddings)

tsne_model = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42) # here, we use t-SNE
tsne_embeddings = tsne_model.fit_transform(embeddings)

df = pd.DataFrame(tsne_embeddings, columns=["x", "y"])
df["Topic"] = topics
df_filtered = df[df["Topic"] != -1]

plt.figure(figsize=(12, 8))
sns.scatterplot(data=df_filtered, x="x", y="y", hue="Topic", palette="tab10", s=60)
plt.title("t-SNE Visualization of BERTopic Clusters (Excluding Outliers)", fontsize=16)
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend(loc='best', bbox_to_anchor=(1.05, 1), title="Topic")
plt.tight_layout()
plt.show()

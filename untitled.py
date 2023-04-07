import csv
import pandas as pd
from fuzzywuzzy import fuzz
from transformers import AutoTokenizer, AutoModel


# Read the product names from the CSV files
amazon_df = pd.read_csv("Amazon.csv")
flipkart_df = pd.read_csv("Flipkart.csv")

# Drop duplicate rows in the dataframes
amazon_df.drop_duplicates(inplace=True)
flipkart_df.drop_duplicates(inplace=True)

# Collect product names from Flipkart and Amazon
flipkart_products = flipkart_df["Name"].tolist()
amazon_products = amazon_df["Name"].tolist()

# Load BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Compute BERT embeddings for Flipkart products
flipkart_embeddings = []
for product in flipkart_products:
    encoded = tokenizer(product, padding=True, truncation=True, return_tensors='pt')
    output = model(**encoded).last_hidden_state.mean(dim=1)
    flipkart_embeddings.append(output.detach().numpy())

# Compute BERT embeddings for Amazon products
amazon_embeddings = []
for product in amazon_products:
    encoded = tokenizer(product, padding=True, truncation=True, return_tensors='pt')
    output = model(**encoded).last_hidden_state.mean(dim=1)
    amazon_embeddings.append(output.detach().numpy())

# Find matches using fuzzy matching
matches = []
for i, flipkart_embedding in enumerate(flipkart_embeddings):
    best_match = None
    best_score = 0
    for j, amazon_embedding in enumerate(amazon_embeddings):
        score = fuzz.ratio(flipkart_products[i], amazon_products[j]) / 100.0
        if score > best_score:
            best_match = amazon_products[j]
            best_score = score
    if best_score >= 0.73:
        matches.append((flipkart_products[i], best_match, best_score))

# Write the matches to a CSV file
with open('matches.csv', mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Flipkart Product', 'Amazon Product', 'Similarity'])
    for match in matches:
        writer.writerow([match[0], match[1], match[2]])
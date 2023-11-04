import csv
import pandas as pd
from fuzzywuzzy import fuzz
from transformers import AutoTokenizer, AutoModel

amazon_df = pd.read_csv("amazon_data.csv")
flipkart_df = pd.read_csv("flipkart_data.csv")

flipkart_products = flipkart_df["product_name"].tolist()
amazon_products = amazon_df["product_name"].tolist()

# Additional attributes
flipkart_colours = flipkart_df["Color"].tolist()
flipkart_control_types = flipkart_df["Control Type"].tolist()
flipkart_weights = flipkart_df["Weight"].tolist()
flipkart_warranties = flipkart_df["Warranty Summary"].tolist()
flipkart_models = flipkart_df["Model Name"].tolist()
flipkart_wattages = flipkart_df["Power Output"].tolist()
flipkart_voltages = flipkart_df["Power Requirement"].tolist()
flipkart_capacities = flipkart_df["Capacity"].tolist()

amazon_colours = amazon_df["Colour"].tolist()
amazon_control_types = amazon_df["Control Type"].tolist()
amazon_weights = amazon_df["Item Weight"].tolist()
amazon_warranties = amazon_df["Warranty"].tolist()
amazon_models = amazon_df["Model"].tolist()
amazon_wattages = amazon_df["Wattage"].tolist()
amazon_voltages = amazon_df["Voltage"].tolist()
amazon_capacities = amazon_df["Capacity"].tolist()

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

flipkart_embeddings = []
for i in range(len(flipkart_products)):
    product = f"{flipkart_products[i]} {flipkart_colours[i]} {flipkart_control_types[i]} {flipkart_weights[i]} {flipkart_warranties[i]} {flipkart_models[i]} {flipkart_wattages[i]} {flipkart_voltages[i]} {flipkart_capacities[i]}"
    encoded = tokenizer(product, padding=True, truncation=True, return_tensors='pt')
    output = model(**encoded).last_hidden_state.mean(dim=1)
    flipkart_embeddings.append(output.detach().numpy())

amazon_embeddings = []
for i in range(len(amazon_products)):
    product_1 = f"{amazon_products[i]} {amazon_colours[i]} {amazon_control_types[i]} {amazon_weights[i]} {amazon_warranties[i]} {amazon_models[i]} {amazon_wattages[i]} {amazon_voltages[i]} {amazon_capacities[i]}"
    encoded_1 = tokenizer(product_1, padding=True, truncation=True, return_tensors='pt')
    output1 = model(**encoded_1).last_hidden_state.mean(dim=1)
    amazon_embeddings.append(output1.detach().numpy())

matches = []
for i, flipkart_embedding in enumerate(flipkart_embeddings):
    best_match = None
    best_score = 0
    for j, amazon_embedding in enumerate(amazon_embeddings):
        score = fuzz.ratio(flipkart_products[i], amazon_products[j]) / 100.0
        if score > best_score:
            best_match = amazon_products[j]
            best_score = score
    if best_score >= 0.85:
        matches.append((flipkart_products[i], best_match, best_score))

with open('matches.csv', mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Flipkart Product', 'Amazon Product', 'Similarity'])
    for match in matches:
        writer.writerow([match[0], match[1], match[2]])


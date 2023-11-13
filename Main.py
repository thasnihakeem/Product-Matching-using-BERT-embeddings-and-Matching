import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Read data
amazon = pd.read_csv("amazon_data.csv")
flipkart = pd.read_csv("flipkart_data.csv")

# Select relevant columns
amazon = amazon[['product_name', 'brand', 'Colour', 'Capacity', 'Model']]
flipkart = flipkart[['product_name', 'brand', 'Color', 'Capacity', 'Model Name']]

# Initialize CountVectorizer
vectorizer = CountVectorizer()

# Function to calculate cosine similarity between two texts
def calculate_similarity(vectorizer, text1, text2):
    return cosine_similarity(vectorizer.transform([text1]), vectorizer.transform([text2]))[0][0]

# Function for product name matching
def product_name_matching(vectorizer, amazon, flipkart):
    product_name_matrix = cosine_similarity(vectorizer.fit_transform(amazon['product_name'].fillna('')),
                                            vectorizer.transform(flipkart['product_name'].fillna('')))
    matching_indices = (product_name_matrix > 0.5).nonzero()
    return matching_indices, product_name_matrix

# Function for brand matching
def brand_matching(vectorizer, amazon, flipkart, matching_product_name_indices):
    matched_brands = []
    for amazon_index, flipkart_index in zip(*matching_product_name_indices):
        brand_similarity = calculate_similarity(vectorizer, amazon.iloc[amazon_index]['brand'], flipkart.iloc[flipkart_index]['brand'])
        if brand_similarity > 0.5:
            matched_brands.append((amazon_index, flipkart_index, brand_similarity))
    return matched_brands

# Function for color matching
def color_matching(vectorizer, amazon, flipkart, matched_brands):
    matched_colors = []
    for amazon_index, flipkart_index, brand_similarity in matched_brands:
        color_similarity = calculate_similarity(vectorizer, str(amazon.iloc[amazon_index]['Colour']),
                                                str(flipkart.iloc[flipkart_index]['Color']))
        if color_similarity > 0.5:
            matched_colors.append((amazon_index, flipkart_index, brand_similarity, color_similarity))
    return matched_colors

# Function for capacity matching
def capacity_matching(amazon, flipkart, matched_colors):
    matched_capacities = []
    for amazon_index, flipkart_index, brand_similarity, color_similarity in matched_colors:
        if amazon.iloc[amazon_index]['Capacity'] == flipkart.iloc[flipkart_index]['Capacity']:
            matched_capacities.append((amazon_index, flipkart_index, brand_similarity, color_similarity))
    return matched_capacities

# Function for model matching
def model_matching(vectorizer, amazon, flipkart, matched_capacities):
    matched_models = []
    for amazon_index, flipkart_index, brand_similarity, color_similarity in matched_capacities:
        model_similarity = calculate_similarity(vectorizer, amazon.iloc[amazon_index]['Model'],
                                                flipkart.iloc[flipkart_index]['Model Name'])
        if model_similarity > 0.7:
            matched_models.append((amazon_index, flipkart_index, brand_similarity, color_similarity, model_similarity))
    return matched_models

# Function to round similarity scores
def round_similarity_score(score, decimals=2):
    return round(score, decimals)


matching_product_name_indices, product_name_matrix = product_name_matching(vectorizer, amazon, flipkart)
matched_brands = brand_matching(vectorizer, amazon, flipkart, matching_product_name_indices)
matched_colors = color_matching(vectorizer, amazon, flipkart, matched_brands)
matched_capacities = capacity_matching(amazon, flipkart, matched_colors)
matched_models = model_matching(vectorizer, amazon, flipkart, matched_capacities)

# Create result DataFrame
columns = ['Amazon_Product_Name', 'Flipkart_Product_Name', 'Product_Name_Similarity', 'Brand_Similarity', 'Color_Similarity', 'Capacity_Similarity', 'Model_Similarity']
result_df = pd.DataFrame(columns=columns)

# Fill result DataFrame
for amazon_index, flipkart_index, brand_similarity, color_similarity, model_similarity in matched_models:
    result_df = result_df.append({'Amazon_Product_Name': amazon.iloc[amazon_index]['product_name'],
                                  'Flipkart_Product_Name': flipkart.iloc[flipkart_index]['product_name'],
                                  'Product_Name_Similarity': round_similarity_score(product_name_matrix[amazon_index, flipkart_index]),
                                  'Brand_Similarity': round_similarity_score(brand_similarity),
                                  'Color_Similarity': round_similarity_score(color_similarity),
                                  'Capacity_Similarity': round_similarity_score(1.0 if amazon.iloc[amazon_index]['Capacity'] == flipkart.iloc[flipkart_index]['Capacity'] else 0.0),
                                  'Model_Similarity': round_similarity_score(model_similarity)}, ignore_index=True)

# Save result DataFrame to CSV
result_df.to_csv("matched_pairs.csv", index=False)

# Print final matched pairs
print("Final Matched Pairs:")
print(result_df[columns])

# Function to find matching products based on user input
def find_matching_products(user_input, matched_models, amazon, flipkart, threshold=0.8):
    filtered_matches = [
        (amazon_idx, flipkart_idx, brand_sim, color_sim, model_sim)
        for amazon_idx, flipkart_idx, brand_sim, color_sim, model_sim in matched_models
        if (user_input.lower() in amazon.iloc[amazon_idx]['product_name'].lower() or
            user_input.lower() in flipkart.iloc[flipkart_idx]['product_name'].lower())
        and model_sim > threshold]

    if filtered_matches:
        print(f"\nMatching products for '{user_input}':")
        for amazon_idx, flipkart_idx, _, _, model_sim in filtered_matches:
            print(f"\nAmazon Product: {amazon.iloc[amazon_idx]['product_name']} - (Similarity: {round_similarity_score(model_sim)})")
            print(f"Flipkart Product: {flipkart.iloc[flipkart_idx]['product_name']} - (Similarity: {round_similarity_score(model_sim)})")
    else:
        print(f"No matching products found for '{user_input}'.")

user_input_product_name = input("Enter a product name from Amazon or Flipkart: ")
find_matching_products(user_input_product_name, matched_models, amazon, flipkart)

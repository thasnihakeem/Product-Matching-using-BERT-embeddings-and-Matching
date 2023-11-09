from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Read data
amazon = pd.read_csv("amazon_data.csv")
flipkart = pd.read_csv("flipkart_data.csv")

# Select relevant columns
amazon = amazon[['product_name', 'brand', 'Colour', 'Capacity', 'Model']]
flipkart = flipkart[['product_name', 'brand', 'Color', 'Capacity', 'Model Name']]

# Initialize CountVectorizer
vectorizer = CountVectorizer()

# Step 1: Product Name Matching
product_name_matrix = cosine_similarity(vectorizer.fit_transform(amazon['product_name'].fillna('')),
                                       vectorizer.transform(flipkart['product_name'].fillna('')))

# Set threshold for product name matching
product_name_threshold = 0.5
matching_product_name_indices = (product_name_matrix > product_name_threshold).nonzero()

# Step 2: Brand Matching
matched_brands = []
for amazon_index, flipkart_index in zip(*matching_product_name_indices):
    brand_similarity = cosine_similarity(vectorizer.transform([amazon.iloc[amazon_index]['brand']]),
                                         vectorizer.transform([flipkart.iloc[flipkart_index]['brand']]))[0][0]
    if brand_similarity > 0.5:
        matched_brands.append((amazon_index, flipkart_index, brand_similarity))

# Step 3: Color Matching
matched_colors = []
color_threshold = 0.5
for amazon_index, flipkart_index, brand_similarity in matched_brands:
    color_similarity = cosine_similarity(vectorizer.transform([str(amazon.iloc[amazon_index]['Colour'])]),
                                         vectorizer.transform([str(flipkart.iloc[flipkart_index]['Color'])]))[0][0]
    if color_similarity > color_threshold:
        matched_colors.append((amazon_index, flipkart_index, brand_similarity, color_similarity))

# Step 4: Capacity Matching
matched_capacities = []
for amazon_index, flipkart_index, brand_similarity, color_similarity in matched_colors:
    if amazon.iloc[amazon_index]['Capacity'] == flipkart.iloc[flipkart_index]['Capacity']:
        matched_capacities.append((amazon_index, flipkart_index, brand_similarity, color_similarity))

# Step 5: Model Matching
matched_models = []
for amazon_index, flipkart_index, brand_similarity, color_similarity in matched_capacities:
    model_similarity = cosine_similarity(vectorizer.transform([amazon.iloc[amazon_index]['Model']]),
                                         vectorizer.transform([flipkart.iloc[flipkart_index]['Model Name']]))[0][0]
    if model_similarity > 0.5:
        matched_models.append((amazon_index, flipkart_index, brand_similarity, color_similarity, model_similarity))

# Create a DataFrame with the specified columns
result_df = pd.DataFrame(columns=['Amazon_Product_Name', 'Flipkart_Product_Name', 'Product_Name_Similarity', 'Brand_Similarity', 'Color_Similarity', 'Model_Similarity'])

# Fill in the DataFrame with the matched pairs and similarity scores
for amazon_index, flipkart_index, brand_similarity, color_similarity, model_similarity in matched_models:
    result_df = result_df.append({
        'Amazon_Product_Name': amazon.iloc[amazon_index]['product_name'],
        'Flipkart_Product_Name': flipkart.iloc[flipkart_index]['product_name'],
        'Product_Name_Similarity': product_name_matrix[amazon_index, flipkart_index],
        'Brand_Similarity': brand_similarity,
        'Color_Similarity': color_similarity,
        'Model_Similarity': model_similarity
    }, ignore_index=True)

# Save the DataFrame to CSV
result_df.to_csv("matched_pairs.csv", index=False)

# Print the final matched pairs
print("Final Matched Pairs:")
print(result_df[['Amazon_Product_Name', 'Flipkart_Product_Name', 'Product_Name_Similarity', 'Brand_Similarity', 'Color_Similarity', 'Model_Similarity']])

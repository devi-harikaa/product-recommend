import pandas as pd
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules

# Step 1: Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('transaction.csv')
    df.columns = df.columns.str.strip()  # Clean up column names
    return df

# Step 2: Preprocess the data to create the basket format (one-hot encoding)
@st.cache_data
def preprocess_data(df):
    # Group data by InvoiceNo and Product, then sum the Quantity
    basket = df.groupby(['InvoiceNo', 'Product'])['Quantity'].sum().unstack().fillna(0)
    
    # Convert Quantity to 1 if the product is purchased, 0 if not (for Market Basket Analysis)
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)
    
    return basket

# Step 3: Apply the Apriori algorithm to find frequent itemsets
def get_frequent_itemsets(basket, min_support=0.05):
    # Apply Apriori algorithm
    frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)
    return frequent_itemsets

# Step 4: Generate Association Rules
def get_association_rules(frequent_itemsets, min_lift=1.0):
    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)
    return rules

# Step 5: Get product recommendations based on association rules
def recommend_product(input_product, basket, rules):
    if input_product not in basket.columns:
        return "Product not found in the dataset"
    
    # Filter the rules where the input product is in the antecedents
    filtered_rules = rules[rules['antecedents'].apply(lambda x: input_product in x)]
    
    # Extract the recommended products from the consequents
    recommended_products = filtered_rules['consequents'].apply(lambda x: list(x)[0]).tolist()
    
    if recommended_products:
        return recommended_products[0]  # Return the best recommendation based on the highest lift
    else:
        return "No recommended product found."

# Step 6: Recommend offers based on the product or category
def recommend_offer(product):
    # Example offers for specific products (extend this with more logic)
    offers = {
        "Milk": "Buy 1 Get 1 Free",
        "Apple": "10% off on your next purchase",
        "Bread": "Free delivery on orders over $20",
        "Banana": "Buy 2 Bananas, Get 1 Free",
        "Chocolate": "10% off on all chocolates",
        "Cereal": "20% off when you buy 2 or more"
    }
    
    # Example category offers (you can categorize products into categories like Dairy, Bakery, Fruits, etc.)
    category_offers = {
        "Dairy": "Free delivery on all dairy products",
        "Fruits": "Buy 2 fruits, Get 1 free",
        "Snacks": "10% off on all snacks this weekend",
        "Bakery": "15% off on all bakery items"
    }
    
    if product in offers:
        return offers[product]
    elif product in ["Milk", "Cheese", "Butter"]:
        return category_offers["Dairy"]
    elif product in ["Apple", "Banana", "Orange"]:
        return category_offers["Fruits"]
    elif product in ["Bread", "Jam"]:
        return category_offers["Bakery"]
    elif product in ["Chocolate", "Cereal"]:
        return category_offers["Snacks"]
    
    return "No special offer available"

# Step 7: Main Streamlit app to interact with the user
def main():
    st.title("Product Recommendation Dashboard")
    
    # Load the dataset
    df = load_data('your_dataset.csv')  # Provide the path to your transaction dataset
    basket = preprocess_data(df)
    
    # Apply Apriori algorithm to find frequent itemsets
    frequent_itemsets = get_frequent_itemsets(basket, min_support=0.05)
    
    # Generate association rules
    rules = get_association_rules(frequent_itemsets, min_lift=1.0)
    
    # Input for the product
    input_product = st.selectbox("Select a product to find recommendations:", basket.columns)
    
    # Get product recommendations based on association rules
    recommended_product = recommend_product(input_product, basket, rules)
    st.write(f"Best product to sell with {input_product}: {recommended_product}")
    
    # Get offer recommendations
    offer = recommend_offer(recommended_product)
    st.write(f"Special offer: {offer}")

if __name__ == "__main__":
    main()

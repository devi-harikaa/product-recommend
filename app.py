import pandas as pd
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules

# Step 1: Load the dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('transaction.csv')  # Ensure the file exists
        df.columns = df.columns.str.strip()  # Clean up column names
        return df
    except FileNotFoundError:
        st.error("Dataset file not found. Please upload 'transaction.csv'.")
        return None

# Step 2: Preprocess the data to create the basket format (one-hot encoding)
@st.cache_data
def preprocess_data(df):
    basket = df.groupby(['InvoiceNo', 'Product'])['Quantity'].sum().unstack().fillna(0)
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)
    return basket

# Step 3: Apply the Apriori algorithm
def get_frequent_itemsets(basket, min_support=0.05):
    frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)
    return frequent_itemsets

# Step 4: Generate Association Rules
def get_association_rules(frequent_itemsets, min_lift=1.0):
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)
    return rules

# Step 5: Recommend Products
def recommend_product(input_product, rules):
    if input_product not in [item for sublist in rules['antecedents'].apply(list) for item in sublist]:
        return "Product not found in rules."
    
    filtered_rules = rules[rules['antecedents'].apply(lambda x: input_product in list(x))]
    if not filtered_rules.empty:
        recommended_products = filtered_rules['consequents'].apply(lambda x: list(x)[0]).tolist()
        return recommended_products[0] if recommended_products else "No recommendations found."
    return "No recommendations found."

# Step 6: Recommend Offers
def recommend_offer(product):
    offers = {
        "Milk": "Buy 1 Get 1 Free",
        "Apple": "10% off on your next purchase",
        "Bread": "Free delivery on orders over $20",
    }
    return offers.get(product, "No special offer available")

# Step 7: Main Streamlit app
def main():
    st.title("Product Recommendation Dashboard")

    # Load the dataset
    df = load_data()
    if df is None:
        st.stop()  # Stop execution if the dataset is not loaded

    basket = preprocess_data(df)
    frequent_itemsets = get_frequent_itemsets(basket)
    rules = get_association_rules(frequent_itemsets)

    # Check if there are any rules
    if rules.empty:
        st.error("No association rules generated. Try reducing the minimum support or lift.")
        st.stop()

    # Product selection
    input_product = st.selectbox("Select a product:", basket.columns)
    if input_product:
        recommended_product = recommend_product(input_product, rules)
        st.write(f"Recommended product to sell with {input_product}: {recommended_product}")

        # Offer recommendation
        offer = recommend_offer(recommended_product)
        st.write(f"Special offer: {offer}")

if __name__ == "__main__":
    main()

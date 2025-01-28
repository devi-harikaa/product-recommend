import pandas as pd
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules
from datetime import datetime

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

# Step 6: Recommend Offers Based on Transaction Percentage and Days Left
def recommend_offer(product, days_left, rules, basket, threshold=0.10):
    # Calculate the percentage of transactions where this product is paired with other products
    total_transactions = len(basket)
    
    # Filter rules where product is in the antecedents
    product_rules = rules[rules['antecedents'].apply(lambda x: product in x)]
    
    # Create a dictionary to store transaction percentages for each product
    product_percentages = {}
    
    for _, row in product_rules.iterrows():
        consequent = list(row['consequents'])[0]
        
        # Filter basket where the product is bought and the consequent is bought
        count_transactions_with_product = len(basket[(basket[product] == 1) & (basket[consequent] == 1)])
        transaction_percentage = count_transactions_with_product / total_transactions
        
        product_percentages[consequent] = transaction_percentage

    # Generate offers based on transaction percentage and expiration days
    offer_message = {}
    
    for recommended_product, percentage in product_percentages.items():
        offer_text = ""
        
        # Apply offer based on transaction percentage
        if percentage >= threshold:
            offer_text += f"Buy {product} and get {recommended_product} at 10% off! "
        else:
            offer_text += f"Get a special offer on {recommended_product} when you buy {product}! "

        # Apply offer based on days left (urgency-based offer)
        if days_left <= 7:
            offer_text += "Hurry, this offer expires soon!"
        elif days_left <= 30:
            offer_text += "Limited time offer! Hurry before it's gone!"
        else:
            offer_text += "Enjoy this offer with no rush."

        offer_message[recommended_product] = offer_text
    
    return offer_message
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
    days_left = st.number_input("Enter the number of days left until expiration:", min_value=0, max_value=365)
    
    if input_product and days_left is not None:
        recommended_product = recommend_product(input_product, rules)
        st.write(f"Recommended product to sell with {input_product}: {recommended_product}")

        # Generate offers based on transaction percentage and expiration days
        offer = recommend_offer(input_product, days_left, rules, basket)
        for product, offer_message in offer.items():
            st.write(f"Special offer for {product}: {offer_message}")

if __name__ == "__main__":
    main()


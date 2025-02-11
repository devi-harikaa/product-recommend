import pandas as pd
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules
import random

# Load and preprocess the data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('transaction.csv')
        df.columns = df.columns.str.strip()
        return df
    except FileNotFoundError:
        st.error("Dataset file not found. Please upload 'transaction.csv'.")
        return None

# Convert transactions into a one-hot encoded basket format
@st.cache_data
def preprocess_data(df):
    basket = df.groupby(['InvoiceNo', 'Product'])['Quantity'].sum().unstack().fillna(0)
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)
    return basket

# Apply the Apriori algorithm
def get_frequent_itemsets(basket, min_support=0.05):
    return apriori(basket, min_support=min_support, use_colnames=True)

# Generate association rules
def get_association_rules(frequent_itemsets, min_lift=1.0):
    return association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)

# Recommend products frequently bought together
def recommend_product(product, rules):
    filtered_rules = rules[rules['antecedents'].apply(lambda x: product in x)]
    recommended = filtered_rules['consequents'].apply(lambda x: list(x)[0]).tolist()
    return recommended[:3] if recommended else ["No strong recommendations"]

# Calculate discount based on days left
def calculate_discount(days_left):
    if days_left > 7:
        return 5  # Low discount
    elif 4 <= days_left <= 7:
        return 10  # Moderate discount
    else:
        return 20  # High discount

# Generate offers based on stock levels and transactions
def recommend_offer(product, df, rules):
    stock_data = df[df['Product'] == product].iloc[-1]
    product_price = stock_data['Price']

    recommended_products = recommend_product(product, rules)
    offer_list = []

    for rec_product in recommended_products:
        if rec_product == "No strong recommendations":
            continue

        rec_data = df[df['Product'] == rec_product].iloc[-1]
        rec_price = rec_data['Price']

        # Dynamically set offer expiry (between 3-10 days)
        offer_days_left = random.randint(3, 10)

        # Calculate discount based on days left
        discount = calculate_discount(offer_days_left)

        offer_list.append({
            "Product": rec_product,
            "Original Price": rec_price,
            "Discount": f"{discount}%",
            "Discounted Price": round(rec_price * (1 - discount / 100), 2)
        })

    return offer_list

# Streamlit UI
def main():
    st.title("Product Recommendations & Offers")

    df = load_data()
    if df is None:
        st.stop()

    basket = preprocess_data(df)
    frequent_itemsets = get_frequent_itemsets(basket)
    rules = get_association_rules(frequent_itemsets)

    if rules.empty:
        st.error("No recommendations found. Try adjusting the parameters.")
        st.stop()

    product = st.selectbox("Select a product:", basket.columns)

    if product:
        offers = recommend_offer(product, df, rules)

        if offers:
            st.markdown(f"### Product: {product}")
            for offer in offers:
                st.markdown(f"""
  - Recommended Product: {offer['Product']}
    - Original Price: ${offer['Original Price']}
    - Discount: {offer['Discount']}
    - Discounted Price: ${offer['Discounted Price']}
                """)
        else:
            st.write("No offers available.")

if __name__ == "__main__":
    main()

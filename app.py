import pandas as pd
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules
import datetime

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

# Calculate dynamic discount based on stock levels
def calculate_discount(stock_on, stock_left):
    stock_sold = stock_on - stock_left
    stock_ratio = stock_sold / stock_on if stock_on > 0 else 0

    # Define discount tiers based on stock movement
    if stock_ratio > 0.8:
        return 5  # Small discount for high-selling products
    elif 0.5 <= stock_ratio <= 0.8:
        return 10  # Moderate discount for steady-selling products
    elif 0.2 <= stock_ratio < 0.5:
        return 15  # Higher discount for slow-moving products
    else:
        return 20  # Maximum discount for stagnant stock

# Generate offers based on stock levels and transactions
def recommend_offer(product, df, rules):
    stock_data = df[df['Product'] == product].iloc[-1]
    stock_left = stock_data['StockLeft']
    total_stock = stock_data['StockOn']
    product_price = stock_data['Price']

    recommended_products = recommend_product(product, rules)
    offer_messages = {}

    for rec_product in recommended_products:
        if rec_product == "No strong recommendations":
            continue

        rec_data = df[df['Product'] == rec_product].iloc[-1]
        rec_price = rec_data['Price']
        rec_stock_on = rec_data['StockOn']
        rec_stock_left = rec_data['StockLeft']

        # Calculate dynamic discount
        discount = calculate_discount(rec_stock_on, rec_stock_left)

        # Dynamically set offer expiry (between 3-10 days)
        offer_days_left = max(3, min(10, rec_stock_left // 5))

        offer_messages[rec_product] = {
            "discount": discount,
            "original_price": rec_price,
            "discounted_price": round(rec_price * (1 - discount / 100), 2),
            "days_left": offer_days_left
        }

    return offer_messages

# Streamlit UI
def main():
    st.title("🛒 Product Recommendation & Offers")

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
            for rec_product, offer_details in offers.items():
                st.markdown(f"""
                **📢 {rec_product} Offer:**
                - 💰 Original Price: **${offer_details['original_price']}**
                - 🔥 Discount: **{offer_details['discount']}%**
                - 🏷️ New Price: **${offer_details['discounted_price']}**
                - ⏳ Offer Expires in: **{offer_details['days_left']} days**
                """)
        else:
            st.write("No offers available.")

if __name__ == "__main__":
    main()

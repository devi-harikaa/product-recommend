import pandas as pd
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules

# Load and preprocess transaction data
@st.cache_data
def load_transactions():
    try:
        df = pd.read_csv('transaction.csv')
        df.columns = df.columns.str.strip()
        return df
    except FileNotFoundError:
        st.error("Transaction dataset not found. Please upload 'transaction.csv'.")
        return None

# Load price and stock data
@st.cache_data
def load_price_stock():
    try:
        df = pd.read_csv('price_stock.csv')
        df.columns = df.columns.str.strip()
        return df
    except FileNotFoundError:
        st.error("Stock dataset not found. Please upload 'price_stock.csv'.")
        return None

# Convert transaction data to basket format
@st.cache_data
def preprocess_data(df):
    basket = df.groupby(['InvoiceNo', 'Product'])['Quantity'].sum().unstack().fillna(0)
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)
    return basket

# Generate frequent itemsets
@st.cache_data
def get_frequent_itemsets(basket, min_support=0.05):
    return apriori(basket, min_support=min_support, use_colnames=True)

# Generate association rules
def get_association_rules(frequent_itemsets, min_lift=1.0):
    return association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)

# Recommend products based on rules
def recommend_product(product, rules):
    product_rules = rules[rules['antecedents'].apply(lambda x: product in x)]
    return product_rules['consequents'].apply(lambda x: list(x)[0]).tolist() if not product_rules.empty else None

# Recommend offers based on stock availability and demand
def recommend_offer(product, days_left, rules, basket, stock_df):
    total_transactions = len(basket)
    product_rules = rules[rules['antecedents'].apply(lambda x: product in x)]
    offers = []

    for _, row in product_rules.iterrows():
        consequent = list(row['consequents'])[0]
        transactions_with_product = len(basket[(basket[product] == 1) & (basket[consequent] == 1)])
        transaction_percentage = transactions_with_product / total_transactions

        stock_info = stock_df[stock_df['Product'] == consequent]
        if stock_info.empty:
            continue

        stock_left = int(stock_info['StockLeft'].values[0])
        price = float(stock_info['Price'].values[0])
        
        discount = 0
        if stock_left < 10:
            discount = 25
        elif days_left <= 7:
            discount = 20
        elif days_left <= 30:
            discount = 15
        elif transaction_percentage >= 0.10:
            discount = 10
        else:
            discount = 5

        offers.append({
            "Product": consequent,
            "Discount": f"{discount}%",
            "Final Price": f"${price * (1 - discount / 100):.2f}"
        })

    return offers

# Streamlit App
def main():
    st.title("Product Recommendation & Offer Generator")
    df = load_transactions()
    stock_df = load_price_stock()
    
    if df is None or stock_df is None:
        st.stop()
    
    basket = preprocess_data(df)
    frequent_itemsets = get_frequent_itemsets(basket)
    rules = get_association_rules(frequent_itemsets)
    
    if rules.empty:
        st.warning("No strong association rules found. Try adjusting the support/lift values.")
        st.stop()
    
    input_product = st.selectbox("Select a product:", basket.columns)
    days_left = st.slider("Days left until expiration:", 0, 365, 30)
    
    if input_product:
        recommendations = recommend_product(input_product, rules)
        if recommendations:
            st.subheader(f"Recommended Products to Sell with {input_product}:")
            st.write(", ".join(recommendations))
        else:
            st.write("No strong product recommendations found.")
        
        offers = recommend_offer(input_product, days_left, rules, basket, stock_df)
        if offers:
            st.subheader("Special Offers:")
            st.table(offers)
        else:
            st.write("No available offers based on current data.")

if __name__ == "__main__":
    main()

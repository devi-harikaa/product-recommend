# Import required libraries
import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.graph_objects as go

# Function to load data
@st.cache
def load_data():
    # Replace with your transactional dataset
    data = pd.read_csv("transactions.csv")  # Update with your dataset file
    return data

# Function to preprocess data for MBA
def preprocess_data(df):
    # Convert transactions into one-hot encoding for Market Basket Analysis
    basket = df.groupby(['InvoiceNo', 'Product'])['Quantity'].sum().unstack().fillna(0)
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)  # Binary representation
    return basket

# Function to perform Market Basket Analysis
def perform_mba(basket, min_support=0.01, metric="lift", min_threshold=1.0):
    frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)
    return rules

# Function to get recommendations for an expiring product
def get_recommendations(rules, product_name, top_n=5):
    recommendations = rules[rules['antecedents'].apply(lambda x: product_name in x)]
    recommendations = recommendations.sort_values(by='lift', ascending=False).head(top_n)
    return recommendations

# Main Streamlit app
def main():
    # Title and description
    st.title("Product Recommendation Dashboard")
    st.markdown("### Provide recommendations for expiring products using Market Basket Analysis (MBA).")
    
    # Load data
    retail_data = load_data()
    st.markdown("### Dataset Overview")
    st.dataframe(retail_data.head())
    
    # Preprocess data
    basket = preprocess_data(retail_data)
    st.markdown("### Processed Basket Data (One-Hot Encoded)")
    st.dataframe(basket.head())

    # Perform MBA
    st.markdown("### Market Basket Analysis Rules")
    rules = perform_mba(basket)
    st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())
    
    # Input: Expiring product
    st.markdown("### Product Recommendation")
    expiring_product = st.text_input("Enter the name of the expiring product:")
    
    if expiring_product:
        recommendations = get_recommendations(rules, expiring_product)
        
        if not recommendations.empty:
            st.markdown(f"### Recommended Products for '{expiring_product}'")
            for _, row in recommendations.iterrows():
                st.markdown(f"- **{', '.join(row['consequents'])}** (Support: {row['support']:.2f}, Confidence: {row['confidence']:.2f}, Lift: {row['lift']:.2f})")
        else:
            st.markdown(f"No recommendations found for the product '{expiring_product}'.")

    # Pie Chart Visualization for Top Products
    st.markdown("### Visualization: Top Products by Frequency")
    top_products = retail_data['Product'].value_counts().head(10)
    fig = go.Figure(data=[go.Pie(labels=top_products.index, values=top_products.values, hole=.3)])
    fig.update_layout(title="Top 10 Products by Sales Frequency")
    st.plotly_chart(fig, use_container_width=True)

# Run the app
if __name__ == "__main__":
    main()

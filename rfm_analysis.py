import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import argparse

def process_data(file_path):
    """
    Load and preprocess the retail data for RFM analysis
    """
    # Load the data
    print("Loading data...")
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    
    # Display basic information
    print(f"Dataset shape: {df.shape}")
    print("\nSample data:")
    print(df.head())
    
    # Check for missing values
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Handle missing CustomerID
    print(f"\nRows with missing CustomerID: {df['CustomerID'].isnull().sum()}")
    df = df.dropna(subset=['CustomerID'])
    print(f"Dataset shape after removing missing CustomerID: {df.shape}")
    
    # Convert CustomerID to integer
    df['CustomerID'] = df['CustomerID'].astype(int)
    
    # Convert InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Calculate total amount for each transaction
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    
    # Filter out canceled orders (indicated by negative quantities or invoice numbers starting with 'C')
    df = df[(df['Quantity'] > 0) & (~df['InvoiceNo'].astype(str).str.startswith('C'))]
    
    print(f"Final dataset shape: {df.shape}")
    return df

def calculate_rfm(df, analysis_date=None):
    """
    Calculate RFM metrics for each customer
    """
    # If no analysis date provided, use the max date in the dataset + 1 day
    if analysis_date is None:
        analysis_date = df['InvoiceDate'].max() + timedelta(days=1)
    else:
        analysis_date = pd.to_datetime(analysis_date)
    
    print(f"\nAnalysis date: {analysis_date}")
    
    # Group by customer and calculate RFM metrics
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (analysis_date - x.max()).days,  # Recency
        'InvoiceNo': 'nunique',                                    # Frequency
        'TotalAmount': 'sum'                                      # Monetary
    }).reset_index()
    
    # Rename columns
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    
    # Validate data
    print("\nRFM metrics summary statistics:")
    print(rfm.describe())
    
    return rfm

def score_rfm(rfm):
    """
    Score and segment customers based on RFM metrics
    """
    # Create quartiles for RFM metrics
    # For Recency, lower values are better (more recent)
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=range(4, 0, -1))
    
    # For Frequency and Monetary, higher values are better
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=range(1, 5))
    rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), 4, labels=range(1, 5))
    
    # Convert to int for easier manipulation
    rfm['R_Score'] = rfm['R_Score'].astype(int)
    rfm['F_Score'] = rfm['F_Score'].astype(int)
    rfm['M_Score'] = rfm['M_Score'].astype(int)
    
    # Calculate RFM combined score (R+F+M)
    rfm['RFM_Score'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']
    
    # Calculate RFM combined string (e.g., 444 for best customers)
    rfm['RFM_String'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
    
    print("\nRFM Score distribution:")
    print(rfm['RFM_Score'].describe())
    
    return rfm

def segment_customers(rfm):
    """
    Assign segments to customers based on RFM scores
    """
    # Define segments based on RFM_Score and individual R, F, M scores
    # These segmentation rules can be adjusted based on business needs
    
    # Create a segment label column
    rfm['Segment'] = ''
    
    # Champions: recent customers who buy often and spend a lot
    rfm.loc[rfm['RFM_Score'] >= 10, 'Segment'] = 'Champions'
    
    # Loyal Customers: buy regularly, recent purchase
    rfm.loc[(rfm['RFM_Score'] >= 8) & (rfm['RFM_Score'] < 10), 'Segment'] = 'Loyal Customers'
    
    # Potential Loyalists: recent customers with average frequency
    rfm.loc[(rfm['R_Score'] >= 3) & (rfm['F_Score'] == 3), 'Segment'] = 'Potential Loyalists'
    
    # New Customers: bought recently but not frequently
    rfm.loc[(rfm['R_Score'] >= 4) & (rfm['F_Score'] == 1), 'Segment'] = 'New Customers'
    
    # Promising: recent shoppers who haven't spent much
    rfm.loc[(rfm['R_Score'] >= 3) & (rfm['F_Score'] == 1) & (rfm['M_Score'] == 1), 'Segment'] = 'Promising'
    
    # Need Attention: above average recency, frequency, and monetary
    rfm.loc[(rfm['R_Score'] == 2) & (rfm['F_Score'] == 2) & (rfm['M_Score'] == 2), 'Segment'] = 'Need Attention'
    
    # About to Sleep: below average recency, frequency, and monetary
    rfm.loc[(rfm['R_Score'] == 2) & (rfm['F_Score'] == 1), 'Segment'] = 'About to Sleep'
    
    # At Risk: haven't purchased for some time, but purchased frequently
    rfm.loc[(rfm['R_Score'] == 1) & (rfm['F_Score'] >= 2), 'Segment'] = 'At Risk'
    
    # Hibernating: last purchase was long ago, low frequency
    rfm.loc[(rfm['R_Score'] == 1) & (rfm['F_Score'] == 1), 'Segment'] = 'Hibernating'
    
    # Assign remaining customers to 'Others'
    rfm.loc[rfm['Segment'] == '', 'Segment'] = 'Others'
    
    # Count customers in each segment
    segment_counts = rfm['Segment'].value_counts().reset_index()
    segment_counts.columns = ['Segment', 'Count']
    
    print("\nCustomer segments:")
    print(segment_counts)
    
    return rfm

def visualize_segments(rfm):
    """
    Create visualizations for customer segments
    """
    plt.figure(figsize=(12, 8))
    
    # Segment distribution pie chart
    plt.subplot(2, 2, 1)
    segments = rfm['Segment'].value_counts()
    plt.pie(segments, labels=segments.index, autopct='%1.1f%%', startangle=90)
    plt.title('Customer Segments Distribution')
    
    # Average RFM values by segment
    segment_rfm = rfm.groupby('Segment').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    }).reset_index()
    
    # Recency by segment (bar chart)
    plt.subplot(2, 2, 2)
    sns.barplot(x='Segment', y='Recency', data=segment_rfm, order=segment_rfm.sort_values('Recency')['Segment'])
    plt.title('Average Recency by Segment')
    plt.xticks(rotation=90)
    
    # Frequency by segment (bar chart)
    plt.subplot(2, 2, 3)
    sns.barplot(x='Segment', y='Frequency', data=segment_rfm, order=segment_rfm.sort_values('Frequency', ascending=False)['Segment'])
    plt.title('Average Frequency by Segment')
    plt.xticks(rotation=90)
    
    # Monetary by segment (bar chart)
    plt.subplot(2, 2, 4)
    sns.barplot(x='Segment', y='Monetary', data=segment_rfm, order=segment_rfm.sort_values('Monetary', ascending=False)['Segment'])
    plt.title('Average Monetary Value by Segment')
    plt.xticks(rotation=90)
    
    plt.tight_layout()
    plt.savefig('rfm_segment_analysis.png')
    print("\nVisualization saved as 'rfm_segment_analysis.png'")
    
    # Scatter plot for Recency vs Frequency colored by Monetary
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        rfm['Recency'], 
        rfm['Frequency'],
        s=rfm['Monetary'] / 100,  # Size based on monetary value
        c=rfm['RFM_Score'],       # Color based on RFM score
        cmap='viridis',
        alpha=0.5
    )
    plt.colorbar(scatter, label='RFM Score')
    plt.xlabel('Recency (days)')
    plt.ylabel('Frequency (# of purchases)')
    plt.title('Customer Segmentation: Recency vs Frequency')
    plt.savefig('rfm_scatter_plot.png')
    print("Visualization saved as 'rfm_scatter_plot.png'")

def generate_segment_insights(rfm):
    """
    Generate business insights and recommendations for each segment
    """
    segment_insights = {
        'Champions': 'Reward them, send personalized offers, ask for reviews, engage them as brand ambassadors',
        'Loyal Customers': 'Upsell higher-value products, recommend loyalty programs, provide personalized service',
        'Potential Loyalists': 'Offer membership or loyalty programs, suggest related products to increase basket size',
        'New Customers': 'Provide onboarding support, educate on products, offer incentives for second purchase',
        'Promising': 'Create awareness about other products, provide early-stage incentives like free shipping',
        'Need Attention': 'Reactivate with limited-time offers, request feedback, recommend popular items',
        'About to Sleep': 'Reactivate with personalized recommendations based on past purchases and limited-time offers',
        'At Risk': 'Send personalized reactivation campaigns, win-back offers, request feedback',
        'Hibernating': 'Reconnect with major discounts, provide new product information, revive interest',
        'Others': 'Monitor for changes in behavior, offer general promotions'
    }
    
    # Summarize segment insights
    print("\nBusiness Insights and Recommendations:")
    for segment, insight in segment_insights.items():
        if segment in rfm['Segment'].unique():
            customer_count = rfm[rfm['Segment'] == segment].shape[0]
            avg_monetary = rfm[rfm['Segment'] == segment]['Monetary'].mean()
            print(f"\n{segment} ({customer_count} customers, Avg. Spend: ${avg_monetary:.2f}):")
            print(f"  - {insight}")

def main():
    parser = argparse.ArgumentParser(description='RFM Customer Segmentation Analysis')
    parser.add_argument('--input', required=True, help='Path to the input CSV file')
    parser.add_argument('--date', required=False, help='Analysis date (YYYY-MM-DD), defaults to max date + 1 day')
    
    args = parser.parse_args()
    
    print("===============================================")
    print("        RFM CUSTOMER SEGMENTATION ANALYSIS     ")
    print("===============================================")
    
    # Process the data
    df = process_data(args.input)
    
    # Calculate RFM metrics
    rfm = calculate_rfm(df, args.date)
    
    # Score RFM
    rfm_scored = score_rfm(rfm)
    
    # Segment customers
    rfm_segmented = segment_customers(rfm_scored)
    
    # Visualize segments
    visualize_segments(rfm_segmented)
    
    # Generate insights
    generate_segment_insights(rfm_segmented)
    
    # Save results
    rfm_segmented.to_csv('customer_rfm_segments.csv', index=False)
    print("\nResults saved to 'customer_rfm_segments.csv'")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()

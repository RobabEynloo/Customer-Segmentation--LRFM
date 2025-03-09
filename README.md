# Customer Segmentation using RFM Analysis

## Overview
This repository contains code for performing customer segmentation using the RFM (Recency, Frequency, Monetary) analysis method on retail transaction data. RFM analysis is a marketing technique used to quantitatively rank and group customers based on their purchasing behavior.

## What is RFM Analysis?
RFM analysis is based on three key metrics:

1. **Recency (R)**: How recently a customer made a purchase. Customers who purchased more recently are more likely to respond to new promotions.
   
2. **Frequency (F)**: How often a customer makes purchases. Customers who purchase frequently are more engaged and likely more loyal.
   
3. **Monetary Value (M)**: How much money a customer spends. Customers who spend more generate more revenue for the business.

By analyzing these three dimensions, businesses can segment their customers into various groups and develop targeted marketing strategies for each segment.

## Dataset Description
The dataset contains retail transaction data with the following columns:
- InvoiceNo: Invoice number (unique to each transaction)
- StockCode: Product code
- Description: Product description
- Quantity: Number of items purchased
- InvoiceDate: Date and time of the transaction
- UnitPrice: Price per unit
- CustomerID: Customer identifier
- Country: Country where the customer resides

## Implementation
The implementation follows these steps:

1. **Data Preprocessing**:
   - Remove canceled orders (those with negative quantities)
   - Handle missing values
   - Calculate total purchase amount for each transaction

2. **RFM Calculation**:
   - Recency: Number of days since a customer's most recent purchase
   - Frequency: Total number of purchases made by each customer
   - Monetary: Total amount spent by each customer

3. **Scoring and Segmentation**:
   - Assign RFM scores to each customer based on percentiles
   - Combine individual scores to create RFM segments
   - Classify customers into meaningful segments like "Champions," "Loyal Customers," "At Risk," etc.

4. **Analysis and Visualization**:
   - Summarize customer segments
   - Visualize the distribution of customers across segments
   - Provide actionable insights based on segmentation

## Potential Business Applications
- Targeted marketing campaigns for different customer segments
- Customer retention strategies for at-risk customers
- Loyalty programs for high-value customers
- Reactivation campaigns for dormant customers
- Cross-selling/up-selling opportunities for loyal customers

## Requirements
- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn (optional, for advanced clustering)

## Usage
Run the Python script with your CSV data file:
```
python rfm_analysis.py --input your_data.csv
```

## Output
- RFM scores and segments for each customer
- Summary statistics for each segment
- Visualizations of customer distribution across segments

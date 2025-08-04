# 🛍️ E-Commerce Customer Segmentation Using RFM and K-Means Clustering

This project analyzes real-world e-commerce customer data to identify distinct customer segments using **RFM (Recency, Frequency, Monetary)** analysis and **K-Means Clustering**. The goal is to help businesses better understand their customer base and tailor marketing strategies accordingly.

---

## 📊 Project Overview

- **Objective**: Segment customers based on their purchasing behavior to drive targeted marketing efforts.
- **Dataset**: Real-world e-commerce transactional data.
- **Tech Stack**: Python, Pandas, Matplotlib, Seaborn, Scikit-learn, Google Colab

---

## 🔍 Dataset Summary

The dataset includes:
- `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`
- Transactions of an online retailer over time
- Location: United Kingdom (primarily)

---

## 📈 Key Techniques

- **Data Preprocessing**
  - Cleaning nulls and duplicates
  - Filtering positive transactions and valid customer IDs

- **RFM Analysis**
  - **Recency**: Days since last purchase
  - **Frequency**: Number of purchases
  - **Monetary**: Total amount spent

- **K-Means Clustering**
  - Optimal clusters determined using the Elbow Method
  - Scaled RFM features for accurate segmentation

- **Data Visualization**
  - Cluster profiling using boxplots and bar charts
  - RFM distribution and interpretation

---

## 👥 Identified Customer Segments

| Cluster | Recency | Frequency | Monetary | Description |
|--------|---------|-----------|----------|-------------|
| 0 | Low | High | High | Loyal, frequent, and high-spending |
| 1 | High | Low | Low | Inactive or churned |
| 2 | Very Low | Very High | Very High | VIP or platinum customers |
| 3 | Medium | Low | Medium | Moderate and occasional buyers |

Each cluster is paired with actionable business strategies such as retention programs, re-engagement plans, and targeted promotions.

---

## 📌 Visualizations

- RFM distribution plots
- Elbow curve for optimal K
- Cluster-wise behavior charts
- Customer value segmentation

---

## 💼 Business Use Cases

- Personalize email campaigns and offers
- Prioritize VIP customer service
- Win back churned or inactive customers
- Optimize marketing budget by focusing on high-value clusters

---

## 📁 Files in Repository

- `customer_segmentation.ipynb` — Main Colab notebook
- `Online Retail.xlsx` — Raw dataset (if publicly shareable)
- `README.md` — Project documentation

---

## 🚀 How to Run

1. Clone the repository  
   ```bash
   git clone https://github.com/snehithmthomas/ecommerce-customer-segmentation.git

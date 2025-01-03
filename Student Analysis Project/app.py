import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit App Configuration
st.set_page_config(
    page_title="Data Science Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Light Theme
st.markdown(
    """
    <style>
    /* Overall App Background and Font */
    .stApp {
        background-color: #f0f0f5;
        color: #262730;
        font-family: "sans-serif";
    }

    /* Button Styling */
    .stButton>button {
        background-color: #6eb52f;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #5a9e2e;
    }

    /* Text Input Styling */
    .stTextInput>input {
        background-color: #e0e0ef;
        color: #262730;
    }

    /* DataFrame Styling */
    .stDataFrame {
        background-color: #ffffff;
        color: #262730;
    }

    /* Markdown Styling */
    .stMarkdown {
        color: #262730;
    }

    /* Sidebar Styling */
    .css-1d391kg { /* Sidebar container */
        background-color: #e0e0ef;
        color: #262730;
    }

    .css-1d391kg a {
        color: #262730;
    }

    .css-1d391kg a:hover {
        color: #6eb52f;
    }

    </style>
    """,
    unsafe_allow_html=True
)



# Define functions for EDA
def load_data():
    """Load dataset into a Pandas DataFrame."""
    st.sidebar.subheader("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    else:
        st.warning("Please upload a CSV file to proceed.")
        return None

def eda_visualizations(df):
    """Generate visualizations for EDA."""
    
    st.subheader("Exploratory Data Analysis")
    st.markdown("---")

    # Correlation Heatmap
    st.write("### Correlation Heatmap")
    corr = df.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
    st.pyplot(plt)

    # Pairplot
    st.write("### Pairplot")
    st.write("This might take time for large datasets.")
    plt.figure()
    sns.pairplot(df)
    st.pyplot(plt)

    # Histogram of Close Prices
    st.write("### Distribution of Close Prices")
    plt.figure(figsize=(10, 6))
    df['Close'].hist(bins=50, alpha=0.7)
    plt.title('Distribution of Close Prices')
    plt.xlabel('Close Price')
    plt.ylabel('Frequency')
    st.pyplot(plt)

    # Scatter Plot: Volume vs Close Price
    st.write("### Scatter Plot: Volume vs Close Price")
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Volume'], df['Close'], alpha=0.5)
    plt.title('Volume vs Close Price')
    plt.xlabel('Volume')
    plt.ylabel('Close Price')
    st.pyplot(plt)

    # Box Plot: Close Prices
    st.write("### Box Plot of Close Prices")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df['Close'])
    plt.title('Box Plot of Close Prices')
    st.pyplot(plt)

    # Missing Value Analysis
    st.write("### Missing Values Analysis")
    missing_data = df.isnull().sum()
    st.write("Missing Values:")
    st.write(missing_data)
    
    st.write("### Missing Data Heatmap")
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Data Heatmap')
    st.pyplot(plt)

    # Box Plot: Outlier Detection in Close Prices
    st.write("### Outlier Detection in Close Prices")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df['Close'])
    plt.title('Outlier Detection in Close Prices')
    st.pyplot(plt)

    # Distribution of Close Prices (Violin Plot)
    st.write("### Distribution of Close Prices (Violin Plot)")
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=df['Close'], color='cyan')
    plt.title('Distribution of Close Prices (Violin Plot)')
    plt.xlabel('Close Price')
    st.pyplot(plt)

    # Plot Close Prices Over Time
    st.write("### Close Prices Over Time")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    plt.figure(figsize=(10, 5))
    plt.plot(df['Timestamp'], df['Close'], alpha=0.6)
    plt.title('Close Prices Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Close Price')
    st.pyplot(plt)

    # Pairplot for Open, High, Low, Close
    st.write("### Pairwise Relationships")
    plt.figure(figsize=(10, 6))
    sns.pairplot(df[['Open', 'High', 'Low', 'Close']])
    plt.suptitle('Pairwise Relationships', y=1.02)
    st.pyplot(plt)

    # Volatility Over Time
    st.write("### Volatility Over Time")
    df['volatility'] = df['High'] - df['Low']
    df['volatility'].plot(title='Bitcoin Price Volatility Over Time')
    st.pyplot(plt)

    # Lag Analysis: Close vs Previous Day's Close
    st.write("### Lag Analysis: Close vs Previous Day's Close")
    df['lagged_close'] = df['Close'].shift(1)
    sns.scatterplot(x=df['lagged_close'], y=df['Close'])
    plt.title('Lag Analysis: Close vs Previous Day\'s Close')
    st.pyplot(plt)

    # Bitcoin Close Price Trend Over Time
    st.write("### Bitcoin Close Price Trend Over Time")
    plt.figure(figsize=(14, 8))
    plt.plot(df.index, df['Close'], label='Close Price', color='green')
    plt.title('Bitcoin Close Price Trend Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    # 30-Day Rolling Average for Close Prices
    df['rolling_close'] = df['Close'].rolling(window=30).mean()
    st.write("### Bitcoin Close Price with 30-Day Rolling Average")
    plt.figure(figsize=(14, 8))
    plt.plot(df.index, df['Close'], label='Close Price', color='green', alpha=0.6)
    plt.plot(df.index, df['rolling_close'], label='30-Day Rolling Average', color='orange', linestyle='--')
    plt.title('Bitcoin Close Price with 30-Day Rolling Average')
    plt.xlabel('Timestamp')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    # Bitcoin Technical Indicator Comparison (Radar Chart)
    st.write("### Bitcoin Technical Indicator Comparison")
    import numpy as np
    labels = ['RSI', 'MACD', 'Moving Avg', 'Volatility', 'Momentum']
    values = [70, 50, 60, 80, 55]
    
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='purple', alpha=0.25)
    ax.plot(angles, values, color='purple', linewidth=2)
    
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    
    plt.title('Bitcoin Technical Indicator Comparison')
    st.pyplot(fig)

    # Bitcoin Price Components Over Time (Stacked Area)
    st.write("### Bitcoin Price Components Over Time (Stacked Area)")
    plt.figure(figsize=(14, 8))
    plt.stackplot(df.index, df['Open'], df['High'], df['Low'], df['Close'],
                labels=['Open', 'High', 'Low', 'Close'], alpha=0.5, colors=['#4e79a7', '#59a14f', '#f28e2b', '#e15759'])
    plt.title('Bitcoin Price Components Over Time (Stacked Area)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend(loc='upper left')
    st.pyplot(plt)

def main():
    st.title("📊 Data Science Dashboard")
    st.markdown("---")
    st.sidebar.title("Navigation")

    # Sidebar options
    options = ["Upload & View Data", "EDA"]
    choice = st.sidebar.radio("Select a page:", options)

    # Page Navigation
    if choice == "Upload & View Data":
        st.header("Upload and View Dataset")
        df = load_data()
        if df is not None:
            st.subheader("Data Overview")
            st.write("Shape of the dataset:", df.shape)
            st.dataframe(df.head())

    elif choice == "EDA":
        st.header("Exploratory Data Analysis")
        df = load_data()
        if df is not None:
            eda_visualizations(df)

if __name__ == "__main__":
    main()

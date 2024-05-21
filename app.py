import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from PIL import Image
import streamlit.components.v1 as components


products = pd.read_csv('product_customers.csv')
category = pd.read_csv('category_stats.csv')

st.set_page_config(layout='wide', initial_sidebar_state='expanded')


# Path to your background image
background_image = 'background.jpg'

# Custom CSS to set the background image
main_bg = f"background-image: url({background_image}); background-size: cover;"
st.markdown(
    f"""
    <style>
    .reportview-container {{
        {main_bg}
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align: center;'>E-Commerce Analytics</h1>", unsafe_allow_html=True)
st.markdown('###')
st.markdown('###')


# Row A
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("<div style='border: 2px solid white; border-radius: 5px; padding: 10px;'>"
                "<p style='color: white;'>Products Sold</p>"
                f"<h3 style='color: white;'>{products['product_id'].nunique()}</h3>"
                "</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div style='border: 2px solid white; border-radius: 5px; padding: 10px;'>"
                "<p style='color: white;'>Customers</p>"
                f"<h3 style='color: white;'>{products['customer_id'].nunique()}</h3>"
                "</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div style='border: 2px solid white; border-radius: 5px; padding: 10px;'>"
                "<p style='color: white;'>Total Revenue</p>"
                f"<h3 style='color: white;'>{round(products['price'].sum(), 2)}</h3>"
                "</div>", unsafe_allow_html=True)

with col4:
    st.markdown("<div style='border: 2px solid white; border-radius: 5px; padding: 10px;'>"
                "<p style='color: white;'>Discounted Amount</p>"
                f"<h3 style='color: white;'>{round(products['discount_amount'].sum(), 2)}</h3>"
                "</div>", unsafe_allow_html=True)

st.markdown('###')
st.markdown('###')

# Row B
c1, c2 = st.columns(2)

with c1:


    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('### Top 10', unsafe_allow_html=True)
    with col3:
        analyse = st.radio('', ['Products', 'Manufacturers'])

    if analyse == 'Products':
        # Get the value counts of product categories
        prod_category_counts = products['product_category'].value_counts()

        # Select the top 10 categories
        prod_top_categories = prod_category_counts.head(10)

        # Filter the dataframe to include only rows with top categories
        prod_top_categories_df = products[products['product_category'].isin(prod_top_categories.index)]

        # Plot the countplot
        plt.figure(figsize=(12, 6), facecolor='none')
        ax = sns.countplot(data=prod_top_categories_df, y='product_category', order=prod_top_categories.index, palette='viridis')
        plt.title('Top Products')

        ax.set_facecolor('black')
        ax.tick_params(axis='y', colors='white')

        plt.xticks([])

        plt.gca().axes.get_xaxis().set_visible(False)

        # Annotate each bar with its count
        for p in ax.patches:
            ax.annotate(f'{int(p.get_width())}', (p.get_width() + 13, p.get_y() + p.get_height() / 2), ha='center', va='center')

        # Display the plot in Streamlit
        st.pyplot(plt)

    elif analyse == 'Manufacturers':
        # Get the value counts of product categories
        manufacturer_counts = products['manufacturer'].value_counts()

        # Select the top 10 categories
        top_manufacturer = manufacturer_counts.head(10)

        # Filter the dataframe to include only rows with top categories
        top_manufacturer_df = products[products['manufacturer'].isin(top_manufacturer.index)]

        # Plot the countplot
        plt.figure(figsize=(12, 6), facecolor='none')
        ax = sns.countplot(data=top_manufacturer_df, y='manufacturer', order=top_manufacturer.index, palette='viridis')
        plt.title('Top Manufacturers')

        ax.set_facecolor('black')
        ax.tick_params(axis='y', colors='white')

        plt.xticks([])

        plt.gca().axes.get_xaxis().set_visible(False)

        # Annotate each bar with its count
        for p in ax.patches:
            ax.annotate(f'{int(p.get_width())}', (p.get_width() + 32, p.get_y() + p.get_height() / 2), ha='center',
                        va='center', color='white')

        st.pyplot(plt)

with c2:
    st.markdown(
        """
        <style>
        .line {
            border-top: 2px solid white;
            margin-top: 20px;
            margin-bottom: 20px;
        }
        </style>
        """
        , unsafe_allow_html=True
    )
    st.markdown(
        """
        <div style='text-align:left'>
            <h3>Basket prediction</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    daily_product_counts = products.groupby('basket_date')['basket_count'].sum().reset_index()

    # Convert 'basket_date' to datetime format
    daily_product_counts['basket_date'] = pd.to_datetime(daily_product_counts['basket_date'])

    # Extract day of the year as integer values for fitting the linear regression model
    daily_product_counts['day_of_year'] = daily_product_counts['basket_date'].dt.dayofyear

    # Fit a linear regression model
    model = LinearRegression()
    model.fit(daily_product_counts[['day_of_year']], daily_product_counts['basket_count'])

    # Predict for the next five days
    future_dates = pd.date_range(start=daily_product_counts['basket_date'].max() + pd.Timedelta(days=1), periods=5)
    future_day_of_year = future_dates.dayofyear
    future_predictions = model.predict(future_day_of_year.values.reshape(-1, 1))

    # Append the future dates and predictions to the existing data for plotting
    all_dates = pd.concat([daily_product_counts['basket_date'], pd.Series(future_dates)])
    all_predictions = np.concatenate((daily_product_counts['basket_count'], future_predictions))

    # Plot the graph
    plt.figure(figsize=(12, 6),facecolor='none')
    ax = plt.gca()
    sns.barplot(x=all_dates.dt.strftime('%Y-%m-%d'), y=all_predictions, color='red', alpha=0.7, label='Predicted')
    sns.barplot(x=daily_product_counts['basket_date'].dt.strftime('%Y-%m-%d'), y=daily_product_counts['basket_count'],
                color='blue', alpha=0.7, label='Actual')

    ax.set_facecolor('black')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='x', colors='white')

    plt.xticks(rotation=90)
    plt.xlabel('Basket Date')
    plt.ylabel('Total Products Count')
    #plt.title('Total Products Added on Each Basket Date with Future Predictions')
    plt.legend()
    st.pyplot(plt)



# Row C
c1, c2 = st.columns(2)

with c1:
    st.markdown('### Category Distribution')
    plt.figure(figsize=(8, 8), facecolor='none')  # Set facecolor to 'none' for transparent background

    # Get unique categories and assign colors
    categories = products['category'].unique()
    colors = sns.color_palette('viridis', len(categories))

    # Pie chart with percentage distribution
    labels = products['category'].value_counts().index
    sizes = products['category'].value_counts().values

    plt.pie(sizes, labels=labels, startangle=90, counterclock=False, wedgeprops=dict(width=0.6), autopct='%1.1f%%',
            textprops={'fontsize': 14, 'color' : 'white'}, colors=colors)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

    # Set the background color of the plot to transparent
    plt.gca().patch.set_alpha(0)

    st.pyplot(plt)
    # Add a border around the column
    st.markdown('<style>div.stDataFrame {border: 2px solid white; border-radius: 5px;}</style>',
                unsafe_allow_html=True)

with c2:
    st.markdown('### Category Statistics')
    st.markdown('###')

    # Your table or other content here
    st.table(category)
    # Add a border around the column
    st.markdown('<style>div.stDataFrame {border: 2px solid white; border-radius: 5px;}</style>', unsafe_allow_html=True)


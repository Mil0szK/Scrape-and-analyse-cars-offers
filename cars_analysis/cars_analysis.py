import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import matplotlib.ticker as ticker
import glob

from docx import Document
from docx.shared import Inches
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.compose import TransformedTargetRegressor


def num_of_models(n):
    segmentation = df.groupby(['model']).size().reset_index(name='offers_number')
    segmentation_sorted=segmentation.sort_values(by='offers_number', ascending=False)
    segmentation_sorted = segmentation_sorted.head(int(n))

    plt.figure(figsize=(12, 8))
    #plt.barh(segmentation_sorted['model'], segmentation_sorted['offers_number'])
    sns.barplot(x='offers_number', y='model', data=segmentation_sorted)
    plt.title('Popularity of cars based on model')
    plt.xlabel('Number of offers')
    plt.ylabel('Model')
    plt.savefig('num_of_models.png', format='png')


def prices_to_time(n):
    print("Choose model from below list containing most popular models: ")
    segmentation = df.groupby(['model']).size().reset_index(name='offers_number')
    segmentation_sorted = segmentation.sort_values(by='offers_number', ascending=False)
    segmentation_sorted = segmentation_sorted.head(int(n))
    print(segmentation_sorted['model'])
    resp = input("Model: ")

    # Filter the dataframe for the chosen model
    df_model = df[df['model'] == resp]

    # Group by year of production and calculate average price
    df_model_avg = df_model.groupby('production')['price'].mean().reset_index()

    plt.figure(figsize=(12, 8))
    # Plot a line plot with year of production on the x-axis and average price on the y-axis
    plt.plot(df_model_avg['production'], df_model_avg['price'])
    plt.title('Average Price vs Year of Production for ' + resp)
    plt.xlabel('Year of Production')
    plt.ylabel('Average Price')
    plt.savefig(f'{resp}.png', format='png')


def prices_to_power():

    df_filtered = df[(df['price'] <= 200000) & (df['power'] <= 300)]

    # Group by power and calculate average price
    df_avg = df_filtered.groupby('power')['price'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(12, 8))
    # Plot a scatter plot with power on the x-axis and average price on the y-axis
    scatter = ax.scatter(df_avg['power'], df_avg['price'])
    ax.set_title('Average Price vs Power for All Cars')
    ax.set_xlabel('Power')
    ax.set_ylabel('Average Price')

    # Format y-axis to display large values with zeros
    formatter = ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x))
    ax.yaxis.set_major_formatter(formatter)

    plt.savefig('prices_to_power.png', format='png')



def correl():
    corelation = df[['mileage', 'price', 'production', 'power', 'capacity']].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corelation, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation')
    plt.savefig('correl.png', format='png')


def cars_location():
    population_data = {
        'Śląskie': 4533565,
        'Opolskie': 986506,
        'Wielkopolskie': 3493969,
        'Zachodniopomorskie': 1701030,
        'Świętokrzyskie': 1241546,
        'Kujawsko-pomorskie': 2077775,
        'Podlaskie': 1181533,
        'Dolnośląskie': 2901225,
        'Podkarpackie': 2129015,
        'Małopolskie': 3400577,
        'Pomorskie': 2333523,
        'Warmińsko-mazurskie': 1428983,
        'Łódzkie': 2466322,
        'Mazowieckie': 5403412,
        'Lubelskie': 2117619,
        'Lubuskie': 1014548
    }
    map_df = gpd.read_file('wojewodztwa.geojson')
    offers_per_province = df['place'].value_counts().reset_index().head(16)
    offers_per_province.columns = ['province', 'offers']
    offers_per_province['population'] = offers_per_province['province'].map(population_data)
    offers_per_province['offers_per_person'] = offers_per_province['offers'] / offers_per_province['population']
    merged = map_df.set_index('nazwa').join(offers_per_province.set_index('province'))

    fig, axs = plt.subplots(2, 1, figsize=(10, 16))

    # Plot the number of offers per province
    merged['offers'] = merged['offers'].fillna(0)
    merged.plot(column='offers', cmap='coolwarm', linewidth=0.8, edgecolor='0.8', legend=True, ax=axs[0])
    axs[0].set_title('Number of offers per province')

    # Plot the number of offers per person in each province
    merged['offers_per_person'] = merged['offers_per_person'].fillna(0)
    merged.plot(column='offers_per_person', cmap='coolwarm', linewidth=0.8, edgecolor='0.8', legend=True, ax=axs[1])
    axs[1].set_title('Number of offers per person in each province')

    plt.tight_layout()
    plt.savefig('cars_location.png', format='png')


def histogram():
    plt.figure(figsize=(12, 8))

    for i, column in enumerate(['price', 'mileage', 'production'], start=1):
        data = df[column]
        lower_limit = np.percentile(data, 5)
        upper_limit = np.percentile(data, 90)
        filtered_data = data[(data >= lower_limit) & (data <= upper_limit)]

        plt.subplot(2, 2, i)
        plt.hist(filtered_data, bins=30, edgecolor='black')
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Number of offers')

    plt.tight_layout()
    plt.savefig('histogram.png', format='png')


def predict_prices():
    # Define features and target variable
    features = ['mileage', 'production', 'power', 'capacity', 'price']
    df_filtered = df[features]

    # Calculate lower and upper limits for each feature and target variable
    for column in df_filtered.columns:
        lower_limit = np.percentile(df[column], 5)
        upper_limit = np.percentile(df[column], 90)
        df_filtered = df_filtered[(df_filtered[column] >= lower_limit) & (df_filtered[column] <= upper_limit)]

    X = df_filtered[['mileage', 'production', 'power', 'capacity']]
    y = np.log(df_filtered['price'])  # Apply logarithmic transformation

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create an instance of the LinearRegression model
    model = LinearRegression()

    # Use TransformedTargetRegressor to apply inverse transformation when predicting
    regr = TransformedTargetRegressor(regressor=model, func=np.log, inverse_func=np.exp)

    # Fit the model with the training data
    regr.fit(X_train, y_train)

    # Use the fitted model to make predictions on the test data
    predictions = regr.predict(X_test)

    # Calculate the mean squared error of the predictions
    mse = mean_squared_error(np.exp(y_test), np.exp(predictions))

    # Calculate the R-squared score
    r2 = r2_score(np.exp(y_test), np.exp(predictions))

    print(f"Mean Squared Error: {mse}")
    print(f"R-squared Score: {r2}")

    # Create a hexbin plot of actual vs predicted prices
    plt.hexbin(np.exp(y_test), np.exp(predictions), gridsize=50, cmap='coolwarm')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs Predicted Prices')
    plt.colorbar(label='Count in Bin')
    plt.savefig('predict_prices.png', format='png')

    return regr


def brand_offers_summary():
    # Define the 30 biggest car brands
    biggest_brands = {'Toyota': 0, 'Volkswagen': 0, 'Ford': 0, 'Honda': 0, 'Chevrolet': 0, 'Nissan': 0, 'Hyundai': 0, 'Mercedes-Benz': 0, 'BMW': 0, 'Audi': 0, 'Lexus': 0, 'Subaru': 0, 'Kia': 0, 'Mazda': 0, 'Jeep': 0, 'Dodge': 0, 'Porsche': 0, 'Land Rover': 0, 'Volvo': 0, 'Jaguar': 0, 'Maserati': 0}

    # Group the dataframe by the 'model' column and count the number of offers for each model
    model_offers = df.groupby('model').size().reset_index(name='offers')

    # For each model, check if it contains a brand from the list of the 30 biggest car brands
    for _, row in model_offers.iterrows():
        for brand in biggest_brands.keys():
            if brand in row['model']:
                biggest_brands[brand] += row['offers']

    # Convert the dictionary to a DataFrame for plotting
    brand_offers = pd.DataFrame(list(biggest_brands.items()), columns=['model', 'offers'])

    # Sort the dataframe by the number of offers in descending order
    brand_offers_sorted = brand_offers.sort_values(by='offers', ascending=False)

    # Create a bar plot for the sorted dataframe
    plt.figure(figsize=(10, 6))
    plt.barh(brand_offers_sorted['model'], brand_offers_sorted['offers'], color='skyblue')
    plt.xlabel('Number of Offers')
    plt.ylabel('Brand')
    plt.title('Number of Offers by Brand')
    plt.gca().invert_yaxis()  # invert the y-axis to display the brand with the most offers at the top

    # Save the plot as a PNG file
    plt.savefig('brand_offers_summary.png', format='png')


def save_plots_to_word():
    # Create a new Word document
    doc = Document()

    svg_files = glob.glob('*.png')
    # Add each plot image to the Word document
    for plot_file in svg_files:
        doc.add_picture(plot_file, width=Inches(6))

    # Save the Word document
    doc.save('car_analysis.docx')


if __name__=="__main__":
    con = sqlite3.connect("cars_base")
    df = pd.read_sql_query("SELECT * FROM cars", con)
    num_of_models(40)
    prices_to_time(20)
    prices_to_power()
    correl()
    cars_location()
    histogram()
    predict_prices()
    brand_offers_summary()
    save_plots_to_word()
    con.close()

# Cars offers analysis
This project is a Python-based application that uses webscraping to get the data about cars offers in Poland
from the otomoto.pl website. The data is then stored in a SQLite database and can be accessed through a Telegram bot.
Then the data is visualized in a variety of ways, and model for predicting the price of a car is created.
That data is then saved to Word document.

## Features

- Web scraping of car offers data from otomoto.pl.
- Storing and managing data in a SQLite database.
- Visualizing data in various ways.
- Creating a model for predicting the price of a car.
- Saving data to a Word document.

## Setup

1. Clone the repository.
2. Install the required dependencies with `pip install -r requirements.txt`.
3. Run the main script with `python cars_analysis.py`.

## Testing

Tests are located in the `tests.py` file. Run them with `python -m unittest tests.py`.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
import re
import sqlite3
from bs4 import BeautifulSoup
from lxml import etree
import requests


def scraping_otomoto(url):
    url_otomoto = url
    page = requests.get(url_otomoto)
    soup = BeautifulSoup(page.text, 'lxml')
    s1 = etree.HTML(str(soup))

    divs = soup.findAll(class_='ooa-10gfd0w e1oqyyyi1')
    cars_data = []

    for div in divs:
        if div.find(class_='eqhgbsk4 ooa-1pzuhke er34gjf0'):
            continue
        info = div.find_all(class_='ooa-1omlbtp e1oqyyyi13')
        model = div.find('h1', class_='e1oqyyyi9 ooa-1ed90th er34gjf0')
        description = div.find('p', class_='e1oqyyyi10 ooa-1tku07r er34gjf0')
        description = description.text.strip().split('•')
        capacity = [i.strip() for i in description if 'cm3' in i]
        power = [j.strip() for j in description if 'KM' in j]

        price = div.find(class_='e1oqyyyi16 ooa-1n2paoq er34gjf0')
        place = div.find('p', class_='ooa-gmxnzj')
        place_re = re.search(r'\((.*?)\)', place.text)
        link = model.find('a')

        if not power:
            power = 0

        if not capacity:
            capacity = 0

        if place_re:
            place = place_re.group(1)
        else:
            place = 'no data'

        mileage = div.find('dd', {'data-parameter': 'mileage'})
        if mileage:
            mileage = mileage.text.strip()
        else:
            mileage = 0

        fuel_type = div.find('dd', {'data-parameter': 'fuel_type'})
        if fuel_type:
            fuel_type = fuel_type.text.strip()
        else:
            fuel_type = 'no data'

        year = div.find('dd', {'data-parameter': 'year'})
        if year:
            year = year.text.strip()
        else:
            year = 0

        gearbox = div.find('dd', {'data-parameter': 'gearbox'})
        if gearbox:
            gearbox = gearbox.text.strip()
        else:
            gearbox = 'no data'

        try:
            car_data = {
                'model': link.text.strip(),
                'capacity': int(''.join(capacity[0].split()[0:-1])) if capacity != 0 else 0,
                'mileage': int(''.join(mileage.split()[0:-1])) if mileage != 0 else 0,
                'price': int(''.join(price.text.strip().split())),
                'gearbox': gearbox,
                'fuel': fuel_type,
                'power': int(''.join(power[0].split()[0:-1])) if power != 0 else 0,
                'production': int(year),
                'place': place
            }

            cars_data.append(car_data)
        except:
            pass
        car_data = {}

        # print(link.text)
        # print(f"{price.text.strip()} PLN")
        # print(''.join(desc for desc in description))
        # for i in info:
        #     print(i.text)
        # print(place.text)
        # print()

    return cars_data


def create_cars_table(c):
    c.execute("DROP TABLE IF EXISTS cars")

    c.execute("""CREATE TABLE IF NOT EXISTS  cars(
        model TEXT,
        capacity INTEGER,
        mileage INTEGER,
        price INTEGER,
        gearbox TEXT,
        fuel TEXT,
        power INTEGER,
        production INTEGER,
        place TEXT)""")


def insert_into_database(cars_data, c):
    for car_data in cars_data:
        c.execute("INSERT INTO cars VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (
            car_data['model'],
            car_data['capacity'],
            car_data['mileage'],
            car_data['price'],
            car_data['gearbox'],
            car_data['fuel'],
            car_data['power'],
            car_data['production'],
            car_data['place']
        ))


def main():
    i = 1
    try:
        conn = sqlite3.connect('cars_base')
        cursor = conn.cursor()
        create_cars_table(cursor)

        for i in range(8000):
            data = scraping_otomoto(f"https://www.otomoto.pl/osobowe?page={i}")
            if not data:
                pass
            insert_into_database(data, cursor)
            print(f"page {i} scraped")
            i += 1

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")

    except requests.RequestException as e:
        print(f"Request error: {e}")

    except Exception as e:
        print(f"An error {e} occurred")

    finally:
        if conn:
            conn.commit()
            conn.close()

if __name__ == '__main__':
    main()
    # conn = sqlite3.connect('cars_base')
    # cursor = conn.cursor()
    # create_cars_table(cursor)
    # data = scraping_otomoto(f"https://www.otomoto.pl/osobowe?page=2")
    # print(data)
    # insert_into_database(data, cursor)


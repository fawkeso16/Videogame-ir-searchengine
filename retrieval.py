# Date : 12 december 2024 
# Author : Oliver Fawkes
# Description : Functions to extract url strings from given csv file, then to scrape relevant data from html and finally to save the data in pickle files. 
# Also includes function to load data from pickle files.
# History : 12/12/2024 - v1 .00, 16/12/2024 - v1 .10


import csv
from bs4 import BeautifulSoup
import pickle
import os
from bs4.element import Comment


# Helper function to clean file paths
def clean_path(path):
    return path.strip().replace('ps2.gamespy.com/', '').replace('videogame/', '')

# Retrieve HTML file paths from CSV 
def read_html_from_csv(csv_path):
    file_paths = []
    with open(csv_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            cleaned_path = clean_path(row[0])
            full_path = f"/Users/oliverfawkes/Downloads/videogames/{cleaned_path}"
            file_paths.append(full_path)
    return file_paths


# Function to process each HTML file, then extract data about the game aswell as a description, then save data in a pickle file.
def process_file_and_save(filepath, directory):
    game_info = {
        "title": None,
        "metadata": {},
        "description": None
    }

    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            html_content = file.read()

        soup = BeautifulSoup(html_content, 'html.parser')
        game_data = soup.find('div', id='content')

        if game_data:
            game_title = soup.find('span', class_='contenttitle')
            if game_title:
                game_info["title"] = game_title.get_text().strip()

            game_metadata = game_data.find_all('td', class_='gameBioInfoText')
            if len(game_metadata) >= 5:
                game_info["metadata"] = {
                    "developer": game_metadata[0].get_text().strip(),
                    "publisher": game_metadata[1].get_text().strip(),
                    "genre": game_metadata[2].get_text().strip(),
                    "releaseDate": game_metadata[3].get_text().strip(),
                    "rating": game_metadata[4].get_text().strip()
                }

            comments = soup.find_all(string=lambda text: isinstance(text, Comment))
            description_text = ""

            for comment in comments:
                if comment.strip() == "DESCRIPTION":
                    current = comment.next_sibling
                    loop_count = 0
                    while current and loop_count < 100:
                        if "#socialBot {text-align:center;margin-top:10px;padding-left:110px;font-size:11px;}" in str(current):
                            break
                        if isinstance(current, Comment) and current.strip() == "/DESCRIPTION":
                            break
                        if isinstance(current, str):
                            description_text += current.strip() + " "
                        prev = current
                        current = current.next_sibling
                        if current == prev:
                            break
                        loop_count += 1

                    if loop_count >= 100:
                        print("Error html read error")

            game_info["description"] = description_text.strip() if description_text else "no desc"

    except FileNotFoundError:
        print(f"File not found: {filepath}")
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")

    # Save the structured data as a pickle file
    if game_info["title"]:  
        pickle_filename = os.path.join(directory, os.path.basename(filepath) + ".pkl")
        with open(pickle_filename, 'wb') as pickle_file:
            pickle.dump(game_info, pickle_file)
        print(f"Data saved to {pickle_filename}")

    return filepath, game_info



#function to load all files from directory that are pickle files
def load_files(directory, num_files):
    pickle_data = {}

    # List all files in the directory
    files = [f for f in os.listdir(directory) if f.endswith('.pkl')]

    files_to_load = files[:num_files]

    for filename in files_to_load:
        file_path = os.path.join(directory, filename)
        try:
            with open(file_path, 'rb') as pickle_file:
                data = pickle.load(pickle_file)
                pickle_data[filename] = data  
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return pickle_data
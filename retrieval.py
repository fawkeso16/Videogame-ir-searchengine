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

# Retrieve HTML file paths from CSV and clean them
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
def process_file_and_save(filepath):
    extracted_text = []

    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            html_content = file.read()

        soup = BeautifulSoup(html_content, 'html.parser')
        game_data = soup.find('div', id='content')

        if game_data:
            #retieve title
            game_title = soup.find('span', class_='contenttitle')
            extracted_text.append(game_title.get_text().strip())
            
            #find data about the game from game info table
            game_metadata = game_data.find_all('td', class_='gameBioInfoText')
            data = [game_metadata[0], game_metadata[1], game_metadata[2], game_metadata[3], game_metadata[4]]
            for info in data:
                extracted_text.append(info.get_text().strip())

            # Collect all #text nodes inside game_data
            comments = soup.find_all(string=lambda text: isinstance(text, Comment))
            description_text = ""

            #extract description, use loop counter to prevent bug where it was infinite looping             
            for comment in comments:
                if comment.strip() == "DESCRIPTION": 
                    current = comment.next_sibling  
                    loop_count = 0  
  
                    while current and loop_count < 100:  # Avoid infinite loops
                        print(f"Loop {loop_count}: Current node: {repr(current)}")
            
                        #to prevent error that was occuring where it wouldnt locate end of description through comments
                        if "#socialBot {text-align:center;margin-top:10px;padding-left:110px;font-size:11px;}" in str(current):
                            break  #

                        #find where description ends
                        if isinstance(current, Comment) and current.strip() == "/DESCRIPTION":
                            break       
                        
                        if isinstance(current, str):
                            description_text += current.strip() + " "
                
                        prev = current
                        current = current.next_sibling
                        if current == prev:
                            break  
                        loop_count += 1  # Increment the safety counter

                    if loop_count >= 100:
                        print("error loading too much info")
         
            if(description_text):
                extracted_text.append(description_text.strip())
            else:
                extracted_text.append("no desc")

    except FileNotFoundError:
        print(f"File not found: {filepath}")
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")

    # Save the extracted data as a pickle file if data was extracted
    if extracted_text:
        pickle_filename = os.path.join("/Users/oliverfawkes/Downloads/videogames", os.path.basename(filepath) + ".pkl")
        with open(pickle_filename, 'wb') as pickle_file:
            pickle.dump(extracted_text, pickle_file)
        print(f"Data saved to {pickle_filename}")
    return filepath 


#function to load all files from directory that are pickle files
def load_files(directory, num_files):
    pickle_data = []

    # List all files in the directory
    files = [f for f in os.listdir(directory) if f.endswith('.pkl')]

    files_to_load = files[:num_files]

    for filename in files_to_load:
        file_path = os.path.join(directory, filename)
        try:
            # Open and load the pickle file
            with open(file_path, 'rb') as pickle_file:
                data = pickle.load(pickle_file)
                pickle_data.append(data)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return pickle_data

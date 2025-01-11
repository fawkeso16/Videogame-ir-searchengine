# Date : 12 december 2024 
# Author : Oliver Fawkes
# Description : Functions to extract url strings from given csv file, then to scrape relevant data from html and finally to save the data in pickle files. 
# Also includes function to load data from pickle files.
# History : 
# 12/12/2024 - v1 .00 - Add data scraping from downloaded html files,formatted text using nlp, added methods to load pickle files.
# 16/12/2024 - v1 .10 - Added better formatting.


import csv
from bs4 import BeautifulSoup
import pickle
import os
from bs4.element import Comment
import re
import spacy 

nlp = spacy.load("en_core_web_sm")  


def clean_path(path):
    return path.strip().replace('/Users/oliverfawkes/Downloads/videogames 2/', '').replace('videogame/', '')

# Retrieve HTML file paths from CSV 

def read_html_from_folder(folder_path):
    file_paths = []
    
    if not os.path.exists(folder_path):
        print("Folder does not exist:", folder_path)
        return file_paths

    for root, dirs, files in os.walk(folder_path):
        print("Inspecting directory:", root)  # Debug: see what os.walk is doing
        for file in files:
            print("Found file:", file)  # Debug: list each file found
            full_path = os.path.join(root, file)
            if file.endswith('.html'):
                file_paths.append(clean_path(full_path))
    
    return file_paths


# Function to process each HTML file, then extract data about the game aswell as a description, then save data in a pickle file.
def process_file_and_save(folder, filepath, output_directory):
    game_info = {
        "title": None,
        "metadata": {},
        "metadata_no_struc": None,
        "description": None
    }

    try:
        with open(folder+ '/' + filepath, 'r', encoding='utf-8') as file:
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
                    "title": game_title.get_text().strip().lower(),
                    "developer": game_metadata[0].get_text().strip().lower(),
                    "publisher": game_metadata[1].get_text().strip().lower(),
                    "genre": game_metadata[2].get_text().strip().lower(),
                    "releaseDate": game_metadata[3].get_text().strip().lower(),
                    "rating": game_metadata[4].get_text().strip().lower()
                }

                game_info["metadata_no_struc"] = [
                    game_title.get_text().strip().lower(),
                    game_metadata[0].get_text().strip().lower(),
                    game_metadata[1].get_text().strip().lower(),
                    game_metadata[2].get_text().strip().lower(),
                    game_metadata[3].get_text().strip().lower(),
                    game_metadata[4].get_text().strip().lower()
                ]

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
                            description_text += current.strip().lower() + " "
                        prev = current
                        current = current.next_sibling
                        if current == prev:
                            break
                        loop_count += 1

                    if loop_count >= 100:
                        print("Error: HTML read error")

            game_info["description"] = description_text.strip() if description_text else "no desc"

    except FileNotFoundError:
        print(f"File not in prcoess found: {folder + '/' + filepath}")
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")

    if game_info["title"]:  
        pickle_filename = os.path.join(output_directory, os.path.basename(filepath) + ".pkl")
        with open(pickle_filename, 'wb') as pickle_file:
            pickle.dump(game_info, pickle_file)
        print(f"Data saved to {pickle_filename}")

    return filepath, game_info

def process_all_files(input_folder, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    file_paths = read_html_from_folder(input_folder)
    for file_path in file_paths:
        process_file_and_save(file_path, output_directory)


def load_files(directory, num_files):
    pickle_data = {}

    files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
    files_to_load = files[:num_files]

    for filename in files_to_load:
        file_path = os.path.join(directory, filename)
        try:
            with open(file_path, 'rb') as pickle_file:
                data = pickle.load(pickle_file)
                pickle_data[filename] = data  
        except FileNotFoundError:
            print(f"File not in load found: {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return pickle_data


def load_and_process_files(directory, num_of_files):
    metadata_dict = {}
    pickle_data = load_files(directory, num_of_files)
    for i, (filename, data) in enumerate(pickle_data.items()):
        if i >= num_of_files:
            print(f"too many {i}")
            break  
        game_info = data  
        base_filename = filename.replace('.html.pkl', '')
        metadata_dict[base_filename] = game_info["metadata"]
    return metadata_dict

def fix_metadata(metadata):
    data = metadata

    step1 = lemmatize_metadata(data)  # Apply lemmatization to the whole structure
    print(step1)
    step2 = clean_metadata(step1)  # Clean the metadata structure
    step3 = clean_developer_and_publisher(step2)  # Clean developer and publisher info

    for game, game_metadata in step3.items():
        # Apply step 4 and 5 to each game's metadata
        step4 = simplify_ratings(game_metadata)  
        step5 = clean_titles(step4)  
        
        step3[game] = step5

    print(step3)
    return step3

def clean_metadata(metadata_dict):
    for file_name, game_metadata in metadata_dict.items():
        for field, field_value in game_metadata.items():
            if isinstance(field_value, str):
                cleaned_value = re.sub(r'[^\w\s]', '', field_value)
                cleaned_value = re.sub(r'[\t\n]', '', cleaned_value)              
                cleaned_value = re.sub(r'\s+', ' ', cleaned_value)                
                cleaned_value = cleaned_value.strip()
                cleaned_value = cleaned_value.lower()              
                game_metadata[field] = cleaned_value

    return metadata_dict

def clean_developer_and_publisher(metadata_dict):
    for filename, metadata in metadata_dict.items():
        for key in ['DEVELOPER', 'PUBLISHER']:
            if key in metadata:
                regex = r'\s*\(.*?\)\s*'  
                cleaned_list = []
                for item in metadata[key]:
                    cleaned_item = re.sub(regex, '', item).strip()
                    cleaned_list.append(cleaned_item)
                metadata[key] = cleaned_list  # Update the metadata dictionary directly
    return metadata_dict

def simplify_ratings(game_metadata):
    rating = game_metadata.get('rating', '')  # Safely access 'rating'
    rating_lower = rating.lower()  # Convert to lowercase for uniformity
    
    if "teen" in rating_lower:
        simplified_rating = "teen"
    elif "everyone" in rating_lower:
        simplified_rating = "everyone"
    elif "mature" in rating_lower:
        simplified_rating = "mature"
    elif "pending" in rating_lower:
        simplified_rating = "rating pending"
    elif "tba" in rating_lower:
        simplified_rating = "tba"
    else:
        simplified_rating = rating_lower
    
    # Update the rating in the game_metadata dictionary
    game_metadata['rating'] = simplified_rating
    return game_metadata


def clean_titles(game_metadata):
    # Access the title and process it
    title = game_metadata.get('title', '')  # Safely access 'title'
    cleaned_title = title.lower()
    cleaned_title = re.sub(r'[^\w\s]', '', cleaned_title)
    cleaned_title = re.sub(r'\s+', ' ', cleaned_title)
    cleaned_title = re.sub(r'\s+s\b', 's', cleaned_title)  # Remove unnecessary 's' at the end
    cleaned_title = cleaned_title.replace('ps2', '')
    # Return the cleaned title and update the game_metadata
    game_metadata['title'] = cleaned_title
    return game_metadata

# For debugging
def lemmatize_metadata(metadata_dict):
    processed_dict = {}
    
    for filename, metadata in metadata_dict.items():
        processed_dict[filename] = {}
        for field, value in metadata.items():
            value = ' '.join(str(value).split())
            doc = nlp(value)
            processed_value = ' '.join([token.lemma_.lower() for token in doc if not token.is_punct])
            processed_dict[filename][field] = processed_value
    
    return processed_dict

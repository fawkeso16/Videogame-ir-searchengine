IR System Overview

This Information Retrieval (IR) system uses TF-IDF weighting to power a basic search engine for a video game depository.
Techniques Implemented:

    Natural Language Processing (NLP)
    TF-IDF Weighting
    Cosine Similarity
    Thesaurus-Based Query Expansion
    Custom Named Entity Recognition (NER)
    Precision@10 Accuracy Testing
    Metadata Boosts

Instructions


The main.py file includes two pre-set methods for testing. Comment out the method you do not wish to run.
Ensure the source code is saved in a folder named Videogame-ir-searchengine. If you rename the folder, update the folder variable in config.py. And please have the 'videogame' data set in your downloads.

    Method 1
        Uses pre-processed pickle files for quick data access.
        Does not create new pickle files.
        No user input is required.

    Method 2
        Runs the full program: scraping HTML data, processing, and storing results.
        Requires specifying the directory for the original video game folder.
        Takes more time to execute compared to Method 1.

Both methods will:

    Output precision/recall results as a graph for the final iteration.
    Display debug statements in the terminal for each query to provide detailed insights.


The main method allows you to test custom queries. Simply comment out the other methods in main.py and enjoy experimenting with the system!


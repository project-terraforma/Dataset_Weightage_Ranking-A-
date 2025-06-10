# Project_A-Unique-Gers-Id
Welcome to the name matcher tool
All the files required to run the application are in the tool folder. Be sure to run the flask_back_end.py and open the link. Also, be sure to download all the required dependencies specified by the requirements.txt

Warning: the tool will download large amounts of data depending on the size of your bounding box. The tools use three model architectures: Xgboost, Sentence Transformers, and Qwen7b

To train these models with your own data refer to the training files folder

Demo: https://youtu.be/T3QkpWgfuJs
Contact: rverma8@ucsc.edu

# Geo-Data Matching and Processing Engine

This project provides a suite of tools to download, process, and match geospatial points of interest (POIs) from OpenStreetMap (OSM) and Overture Maps.

The backend is built with Flask and includes a data downloader, a sophisticated matching script using fuzzy logic, sentence-transformers, and an LLM, and endpoints to manage the process.

## License

The code in this project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Data Source Attribution

This project utilizes data from multiple open sources. The following attributions are required:

* **Overture Maps**: The application downloads and processes data from Overture Maps.
    > This data is available under the CDLA-Permissive-2.0 license and Overture Maps data is available under the ODbL license. Copyright © Overture Maps Foundation.

* **OpenStreetMap**: The application downloads and processes data from OpenStreetMap via OSMnx and the Overpass API.
    > This data is made available under the Open Database License (ODbL). © OpenStreetMap contributors.

import requests
import time
import os
import pandas as pd
import socks
import socket

# fyi i used a proxy to avoid getting banned by the server
# you can use your own proxy or remove this part if you don't need it
socks.set_default_proxy(socks.SOCKS5, "host", port, username="user", password="pwd")
socket.socket = socks.socksocket

def download_page(title, url, output_dir = None):
	# you would usually use a header, but we use rotating proxies so no need
	
	file_already_exists = False # setting this outside the try-except block avoids errors
	
	try:
		if output_dir and not os.path.exists(output_dir):
			os.makedirs(output_dir)

		# this fixes the title, otherwise we can't read the csv
		fixed_title = title.replace("|", ",")
		fixed_title = fixed_title.replace("}", "-")
		filepath = os.path.join(output_dir, fixed_title + ".pdf")

		if not os.path.exists(output_dir):
			os.makedirs(output_dir)

		file_already_exists = os.path.exists(filepath)

		if file_already_exists:
			print(f"File {filepath} already exists. Skipping download.")
			return
		else: # handle the request only if the file isn't downloaded already -> don't overload server
			response = requests.get(url, stream=True)  # Fetch the HTML
			response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

		print(f"Downloading {url} to {filepath}...")
		response = requests.get(url, stream=True)
		response.raise_for_status()

		# not sure if this really is needed
		with open(filepath, "wb") as f:
			for chunk in response.iter_content(chunk_size=8192):
				f.write(chunk)

		print(f"Downloaded {fixed_title} successfully!")

	except requests.exceptions.RequestException as e:
		print(f"Error downloading {url}: {e}")
		return
	except Exception as e:
		print(f"An unexpected error occurred: {e}")
		return
	finally:
		if not file_already_exists:
			time.sleep(0) # sleep for 15 second to avoid overloading the server !! no proxy = 15s
			
def process_csv(csv_filepath): # this helper function just reads the csv and calls the download_page function
	"""Processes a CSV file containing arXiv URLs.

	Args:
		csv_filepath: The path to the CSV file.
	"""
	try:
		df = pd.read_csv(csv_filepath)
		for row in df.itertuples():
			title, authors, date, url = row[1], row[2], row[3], row[4]
			download_page(title, url, output_dir = os.path.join("papers/", str(date) + "/"))
	except Exception as e:
		print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    csv_file = "python/fixed_arxiv_links.csv" # replace with own csv path to the processed files
    process_csv(csv_file)

# EOF
import requests
import time
import os

def download_page(title, authors, year, url, output_dir = None):
	headers = {
        "User-Agent": "Script to download quantum computing arxiv papers (example@student.maastrichtuniversity.nl)" # replace with own email, this isn't mandatory but it's nice to let the server admin know who you are
    }

	# Default flag to False
	file_already_exists = False
	
	try:		
		if output_dir and not os.path.exists(output_dir):
			os.makedirs(output_dir)

		# Sanitize filename
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
			response = requests.get(url, headers=headers, stream=True)  # Fetch the HTML
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
			time.sleep(15)  # change this at own risk, this is basically the time between requests to the server as per robots.txt guidelines
# EOF
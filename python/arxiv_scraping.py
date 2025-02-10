import requests
import time

def download_page(title, authors, year, url, output_dir = None):
	headers = {
        "User-Agent": "Script to download quantum computing arxiv papers (example@student.maastrichtuniversity.nl)" # replace with own email, this isn't mandatory but it's nice to let the server admin know who you are
    }
	
	try:
		# get PDF, handle the longer links with bs4 -- mostly Gemini logic here
		response = requests.get(url, headers=headers, stream=True)  # Fetch the HTML
		response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
		
		fixed_title = title.replace("|", ",")  # replace | with commas
		
		filepath = os.path.join(output_dir, fixed_title + ".pdf")

		if not os.path.exists(output_dir):
			os.makedirs(output_dir) 

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
	except Exception as e:
		print(f"An unexpected error occurred: {e}")
	finally:
		time.sleep(15)  # change this at own risk, this is basically the time between requests to the server as per robots.txt guidelines
		
def process_csv(csv_filepath):
	"""Processes a CSV file containing arXiv URLs.

	Args:
		csv_filepath: The path to the CSV file.
	"""
	try:
		df = pd.read_csv(csv_filepath)
		for row in df.itertuples():
			title, authors, date, url = row[1], row[2], row[3], row[4]
			download_page(title, authors, date, url, output_dir="papers/")
	except Exception as e:
		print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    csv_file = "fixed_arxiv_links.csv"  # replace with own csv path
    process_csv(csv_file)

# EOF
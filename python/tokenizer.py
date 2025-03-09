from pdfminer.high_level import extract_text
import re
import os

def tokenize_text(dir):
	"""Tokenizes text from PDF files in a directory.

	Args:
		dir: The directory containing the PDF files.
	"""
	pdf_files = [f for f in os.listdir(dir) if f.endswith(".pdf")]
	for pdf_file in pdf_files:
		pdf_path = os.path.join(dir, pdf_file)
		txt_path = pdf_path.replace(".pdf", ".txt")
		if os.path.exists(txt_path):
			print(f"skipping {pdf_path}, text file already exists / pdf already tokenized.")
			continue
		try:
			text = extract_text(pdf_path)
		except Exception as e:
			print(f"Error extracting text from {pdf_path}: {e}")
			continue
		text = re.sub(r'[^a-zA-Z]', ' ', text) # remove non-alphabetic characters
		text = re.sub(r'<.*?>', '', text) # remove angled brackets
		text = re.sub(r'[^\w\s]', '', text)
		text = re.sub(r'\n', '', text) # remove new line characters from the text
		text = re.sub(r'\d', '', text) # remove digits
		text = re.sub(r'[\|#-]', '', text) # remove special characters
		text = re.sub(r'\b[a-zA-Z]\b', '', text) # remove single characters
		text = re.sub(r'\s+', ' ', text).strip() # remove extra whitespaces
		stop_words = {"the", "is", "a", "an", "of", "in", "on", "at", "to", "and", "or", "it"}
		text = " ".join([word for word in text.split() if word.lower() not in stop_words])
		text = text.lower()

		with open (pdf_path.replace(".pdf", ".txt"), "w") as f:
			f.write(text)
		print(f"tokenized {pdf_path} to {txt_path}")

if __name__ == "__main__":
	for year in os.listdir("/Users/lpaggen/Documents/DACS_COURSES/dsdm_research_sem2/papers/"):
		if year != ".DS_Store":
			tokenize_text(f"/Users/lpaggen/Documents/DACS_COURSES/dsdm_research_sem2/papers/{year}")
# EOF 
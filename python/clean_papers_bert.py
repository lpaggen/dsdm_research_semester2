import re
import os
import shutil

def process_papers(classified_dir, text_data_dir, output_dir):
    # this function is responsible for getting the text from a file
    def extract_intro_to_references(text):
        intro_pattern = re.compile(
            r'^\s*(\d+\.?\s*|[IVX]+\.\s*)?INTRODUCTION\s*$',
            re.IGNORECASE | re.MULTILINE
        )
        references_pattern = re.compile(r'\[\s*1\s*\]\s*[A-Za-z]\.')
        intro_match = intro_pattern.search(text)
        start_pos = intro_match.start() if intro_match else 0
        references_match = references_pattern.search(text, start_pos)
        if not references_match:
            return None
        end_pos = references_match.start()
        return text[start_pos:end_pos]

    # this removes short lines from the text because they often contain equations and artifacts
    def remove_short_lines(text, min_chars=25):
        return "\n".join(line for line in text.splitlines() if len(line.strip()) >= min_chars)

    def keep_letters_and_basic_punct(text):
        cleaned = re.sub(r'[^a-zA-Z0-9.,!?\s]', '', text)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()

    def remove_cid_substrings(text):
        return text.replace("cid", "")

    # some last cleaning steps
    def clean_text_pipeline(text):
        text = remove_cid_substrings(text)
        text = re.sub(r'[,\d]', '', text)
        text = re.sub(r'(\s*\.\s*){2,}', '.', text)
        text = re.sub(r'\.+', '.', text)
        text = re.sub(r'\b([b-hj-zB-HJ-Z])\b', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    for year in os.listdir(classified_dir):
        if year == ".DS_Store":
            continue
        year_path = os.path.join(classified_dir, year)
        for paper in os.listdir(year_path):
            if not paper.endswith(".txt"):
                continue
            file_path = os.path.join(year_path, paper)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            extracted = extract_intro_to_references(text)
            if not extracted:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write("")
                print(f"No intro-to-references found in {paper}, file cleared.")
                continue
            cleaned_text = remove_short_lines(extracted, min_chars=15)
            cleaned_text = keep_letters_and_basic_punct(cleaned_text)
            cleaned_text = clean_text_pipeline(cleaned_text)
            with open(file_path, "w", encoding="utf-8") as f:
                if cleaned_text.strip():
                    f.write(cleaned_text)
                    print(f"Processed and overwritten {paper}.")
                else:
                    f.write("")
                    print(f"{paper} cleaned to empty, file cleared.")

    for year in os.listdir(text_data_dir):
        if year == ".DS_Store":
            continue
        path = os.path.join(text_data_dir, year)
        for paper in os.listdir(path):
            if paper.endswith(".txt"):
                file_path = os.path.join(path, paper)
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                if not text.strip(): # also skip files with only whitespace
                    os.remove(file_path)

    os.makedirs(output_dir, exist_ok=True)
    file_counter = 0

    for year in os.listdir(text_data_dir):
        if year == ".DS_Store":
            continue
        year_path = os.path.join(text_data_dir, year)
        if not os.path.isdir(year_path):
            continue
        for paper in os.listdir(year_path):
            if not paper.endswith(".txt"):
                continue
            source_file = os.path.join(year_path, paper)
            new_filename = f"{year}_{paper}"
            target_file = os.path.join(output_dir, new_filename)
            while os.path.exists(target_file):
                file_counter += 1
                new_filename = f"{year}_{file_counter}_{paper}"
                target_file = os.path.join(output_dir, new_filename)
            shutil.copy2(source_file, target_file)

if __name__ == "__main__":
    classified_dir = "classified_papers"
    text_data_dir = "text_data"
    output_dir = "processed_papers"
    
    process_papers(classified_dir, text_data_dir, output_dir)
    print("Processing complete. All files have been cleaned and copied to the output directory.")

import csv
import re

def fix_commas_before_author(row_string):
    """Replaces non-delimiting commas in each row with semicolons."""
    match = re.search(r'(.*)(,)([^,]+,\d{4},.*)', row_string) # this is Gemini, the regex is a bit too complex otherwise
    if match:
        title_part = match.group(1)
        author_date_url_part = match.group(3)
        modified_title = title_part.replace(",", "|").replace("/", "~") # here we need to replace the commas in the title otherwise pandas won't be able to open the file in CSV format
        # modified_title = title_part.replace("/", "}")
        return modified_title + "," + author_date_url_part
    else:
        return row_string

def process_csv(input_filepath, output_filepath):
    """Processes the CSV file, fixing commas in titles."""
    try:
        with open(input_filepath, 'r', encoding='utf-8') as infile, \
                open(output_filepath, 'w', newline='', encoding='utf-8') as outfile:

            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            for row in reader:
                row_string = ",".join(row)  # convert row to string
                modified_row_string = fix_commas_before_author(row_string)
                modified_row = modified_row_string.split(",") # convert string back to row
                writer.writerow(modified_row)

    except FileNotFoundError:
        print(f"Input file not found: {input_filepath}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    input_csv = "papers.csv" # replace with path to csv
    output_csv = "fixed_arxiv_links.csv"
    process_csv(input_csv, output_csv)
    print(f"Fixed CSV file saved to: {output_csv}")
# EOF
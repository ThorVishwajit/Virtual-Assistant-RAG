import json

# --- Configuration ---
# The name of your original dataset file
input_file_name = 'brand.json'
# The name of the new file that will be created
output_file_name = 'mikasa_rag_dataset2.jsonl'

print(f"Starting conversion of '{input_file_name}'...")

try:
    # Open the input file to read and the output file to write
    with open(input_file_name, 'r', encoding='utf-8') as infile, \
         open(output_file_name, 'w', encoding='utf-8') as outfile:
        
        # Load the entire JSON array from the input file
        data = json.load(infile)
        
        # Loop through each conversation pair in the list
        for entry in data:
            # Create a new dictionary with the keys your RAG engine needs ("instruction", "output")
            new_entry = {
                "instruction": entry["user"],
                "output": entry["bot"]
            }
            # Write the new dictionary as a single, separate line in the output file
            outfile.write(json.dumps(new_entry) + '\n')

    print(f"✅ Success! Conversion complete.")
    print(f"   New file created: '{output_file_name}'")

except FileNotFoundError:
    print(f"❌ Error: The input file '{input_file_name}' was not found.")
    print("   Please make sure it's in the same folder as this script.")
except Exception as e:
    print(f"An error occurred: {e}")
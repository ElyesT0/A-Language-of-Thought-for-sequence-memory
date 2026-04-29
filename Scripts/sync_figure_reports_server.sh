#!/bin/bash

# Step 1: Convert Jupyter Notebook to HTML without input cells
NOTEBOOK_NAME="Sequence_learning_article_figures.ipynb"
OUTPUT_FORMAT="html"
OUTPUT_FILE="Sequence_learning_article_figures.html"

echo "Converting Jupyter Notebook ($NOTEBOOK_NAME) to HTML..."
jupyter nbconvert $NOTEBOOK_NAME --to $OUTPUT_FORMAT --no-input

# Check if the conversion was successful
if [[ $? -ne 0 ]]; then
  echo "Error: Jupyter Notebook conversion failed."
  exit 1
fi

echo "Conversion successful. HTML file generated: $OUTPUT_FILE"

# Step 2: Sync the generated HTML file to the remote server
REMOTE_USER="ubuntu"
REMOTE_HOST="141.94.79.48"
REMOTE_PORT="57688"
REMOTE_PATH="/var/www/html/reports/memocrush"

echo "Syncing the HTML file to the remote server..."
rsync -avP -e "ssh -p $REMOTE_PORT" $OUTPUT_FILE $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH

# Check if the rsync command was successful
if [[ $? -ne 0 ]]; then
  echo "Error: File synchronization failed."
  exit 1
fi

echo "File synchronization successful."
echo "Process completed."


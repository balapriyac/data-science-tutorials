#!/bin/bash

# Define input and output files
INPUT_FILE="users.csv"
OUTPUT_FILE="users_cleaned.csv"
TEMP_FILE="temp.csv"

echo "Starting data cleaning process..."

# Step 1: Handle Missing Values
echo "Step 1: Handling missing values..."
sed 's/,,/,NULL,/g; s/,$/,NULL/g' $INPUT_FILE > $OUTPUT_FILE

# Step 2: Fix Missing First Names
echo "Step 2: Fixing missing first names..."
awk -F, 'BEGIN {OFS=","} {if ($2 == "" || $2 == "NULL") $2 = "Unknown"; print}' $OUTPUT_FILE > $TEMP_FILE
mv $TEMP_FILE $OUTPUT_FILE

# Step 3: Fix Invalid Email Formats
echo "Step 3: Fixing invalid email formats..."
awk -F, 'BEGIN {OFS=","} {if ($3 !~ /@/ || $3 == "" || $3 == "NULL" || $3 == "not_an_email") $3 = "unknown@example.com"; print}' $OUTPUT_FILE > $TEMP_FILE
mv $TEMP_FILE $OUTPUT_FILE

# Step 4: Correct Date Formats
echo "Step 4: Correcting date formats..."
awk -F, 'BEGIN {OFS=","} {if ($5 == "invalid_date" || $5 == "" || $5 == "NULL") $5 = "2023-01-20"; print}' $OUTPUT_FILE > $TEMP_FILE
mv $TEMP_FILE $OUTPUT_FILE

# Step 5: Ensure Last Login Date is Valid
echo "Step 5: Ensuring last login date is valid..."
awk -F, 'BEGIN {OFS=","} {if ($6 == "" || $6 == "NULL") $6 = "2023-03-23"; print}' $OUTPUT_FILE > $TEMP_FILE
mv $TEMP_FILE $OUTPUT_FILE

# Step 6: Handle Negative Values
echo "Step 6: Handling negative values..."
awk -F, 'BEGIN {OFS=","} {if ($7 < 0) $7 = 0; print}' $OUTPUT_FILE > $TEMP_FILE
mv $TEMP_FILE $OUTPUT_FILE

# Validation checks
echo "Running validation checks..."

# Check for empty fields
EMPTY_FIELDS=$(grep -c ",," $OUTPUT_FILE)
echo "Empty fields remaining: $EMPTY_FIELDS"

# Check for invalid emails
INVALID_EMAILS=$(grep -v -E '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}' $OUTPUT_FILE | grep -v "email" | wc -l)
echo "Invalid emails remaining: $INVALID_EMAILS"

# Check for invalid dates
INVALID_DATES=$(grep -v -E '[0-9]{4}-[0-9]{2}-[0-9]{2}' $OUTPUT_FILE | grep -v "signup_date" | wc -l)
echo "Invalid dates remaining: $INVALID_DATES"

# Check for negative values
NEGATIVE_VALUES=$(awk -F, '$7 < 0 {print}' $OUTPUT_FILE | wc -l)
echo "Negative values remaining: $NEGATIVE_VALUES"

echo "Data cleaning complete. Cleaned data saved to $OUTPUT_FILE"

# Optional: Remove temporary file if it exists
if [ -f "$TEMP_FILE" ]; then
    rm $TEMP_FILE
fi

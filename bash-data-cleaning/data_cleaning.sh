#!/bin/bash

# Data Cleaning at the Command Line - Complete Script
# This script demonstrates various command-line data cleaning techniques

echo "========================================="
echo "Data Cleaning Command-Line Tutorial"
echo "========================================="
echo ""

# Step 1: Create sample messy data
echo "Step 1: Creating sample messy dataset..."
cat > messy_data.csv << 'EOF'
name,age,salary,department,email
John Doe,32,50000,Engineering,john@example.com
Jane Smith,28,55000,Marketing,jane@example.com
   Bob Johnson   ,35,60000,Engineering,bob@example.com
Alice Williams,29,,Marketing,alice@example.com
Charlie Brown,45,70000,Sales,charlie@example.com
Dave Wilson,31,52000,Engineering,
Emma Davis,,58000,Marketing,emma@example.com
Frank Miller,38,65000,Sales,frank@example.com
John Doe,32,50000,Engineering,john@example.com
Grace Lee,27,51000,Engineering,grace@example.com
EOF
echo "✓ Sample data created: messy_data.csv"
echo ""

# Step 2: Explore the data
echo "Step 2: Exploring the data..."
echo "First 5 rows:"
head -n 5 messy_data.csv
echo ""
echo "Last 3 rows:"
tail -n 3 messy_data.csv
echo ""
echo "Total row count:"
wc -l messy_data.csv
echo ""

# Step 3: View specific columns
echo "Step 3: Viewing specific columns (name, department)..."
cut -d',' -f1,4 messy_data.csv
echo ""

# Step 4: Remove duplicates
echo "Step 4: Removing duplicate rows..."
head -n 1 messy_data.csv > cleaned_data.csv
tail -n +2 messy_data.csv | sort | uniq >> cleaned_data.csv
echo "✓ Duplicates removed, saved to: cleaned_data.csv"
echo ""

# Step 5: Search and filter with grep
echo "Step 5: Searching and filtering..."
echo "All engineers:"
grep "Engineering" messy_data.csv
echo ""
echo "Rows with missing data (empty fields):"
grep ",," messy_data.csv
echo ""
echo "Excluding rows with missing data..."
grep -v ",," messy_data.csv > no_missing.csv
echo "✓ Saved to: no_missing.csv"
echo ""

# Step 6: Trim whitespace
echo "Step 6: Trimming whitespace..."
sed 's/^[ \t]*//; s/[ \t]*$//' messy_data.csv > trimmed_data.csv
echo "✓ Whitespace trimmed, saved to: trimmed_data.csv"
echo ""

# Step 7: Replace values
echo "Step 7: Replacing values..."
echo "Replacing 'Engineering' with 'Tech':"
sed 's/Engineering/Tech/g' messy_data.csv | head -n 5
echo ""
echo "Filling empty email fields with default:"
sed 's/,$/,no-email@example.com/' messy_data.csv | grep "no-email"
echo ""

# Step 8: Count and summarize with awk
echo "Step 8: Counting and summarizing..."
echo "Records by department:"
tail -n +2 messy_data.csv | cut -d',' -f4 | sort | uniq -c
echo ""
echo "Average age calculation:"
tail -n +2 messy_data.csv | awk -F',' '{if($2) sum+=$2; if($2) count++} END {print "Average age:", sum/count}'
echo ""

# Step 9: Combining commands with pipes
echo "Step 9: Combining commands..."
echo "Unique departments (sorted):"
tail -n +2 messy_data.csv | cut -d',' -f4 | sort | uniq
echo ""
echo "Engineers with salary > 55000:"
tail -n +2 messy_data.csv | grep "Engineering" | awk -F',' '$3 > 55000' | cut -d',' -f1,3
echo ""
echo "Employee count per department (sorted by count):"
tail -n +2 messy_data.csv | cut -d',' -f4 | sort | uniq -c | sort -rn
echo ""

# Step 10: Convert data formats
echo "Step 10: Converting data formats..."
echo "Converting CSV to TSV..."
sed 's/,/\t/g' messy_data.csv > data.tsv
echo "✓ Saved to: data.tsv"
echo ""
echo "Adding year column..."
awk -F',' 'BEGIN{OFS=","} {print $0, "2024"}' messy_data.csv > data_with_year.csv
echo "✓ Saved to: data_with_year.csv"
echo "Preview:"
head -n 3 data_with_year.csv
echo ""

# Step 11: Complete cleaning pipeline
echo "Step 11: Running complete cleaning pipeline..."
head -n 1 messy_data.csv > final_clean.csv
tail -n +2 messy_data.csv | \
  sed 's/^[ \t]*//; s/[ \t]*$//' | \
  grep -v ",," | \
  sort | \
  uniq >> final_clean.csv
echo "✓ Final cleaned data saved to: final_clean.csv"
echo ""
echo "Final cleaned data preview:"
cat final_clean.csv
echo ""

# Summary
echo "========================================="
echo "Summary of generated files:"
echo "========================================="
echo "- messy_data.csv        : Original messy dataset"
echo "- cleaned_data.csv      : Duplicates removed"
echo "- no_missing.csv        : Rows with missing data excluded"
echo "- trimmed_data.csv      : Whitespace trimmed"
echo "- data.tsv              : Tab-separated format"
echo "- data_with_year.csv    : With year column added"
echo "- final_clean.csv       : Fully cleaned (best version)"
echo ""
echo "Cleaning complete!"


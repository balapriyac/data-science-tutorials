
# Command Line Statistics Script – Usage Guide

This guide demonstrates is a quick reference to use the **`command_line_stats.sh`** Bash script to perform statistical analysis on CSV data directly from the command line.

---

## 1. Setup

Make sure the script is executable:

```bash
chmod +x command_line_stats.sh
```

Create the sample dataset:

```bash
./command_line_stats.sh create_data
```

This will generate a `traffic.csv` file with sample website traffic data.

---

## 2. Exploring the Dataset

### Count rows in the CSV:

```bash
./command_line_stats.sh count_rows
```

---

### View the first 5 rows:

```bash
./command_line_stats.sh view_head
```

---

### Extract the visitors column:

```bash
./command_line_stats.sh extract_visitors
```

---

## 3. Measures of Central Tendency

### Calculate the mean:

```bash
./command_line_stats.sh mean
```

---

### Calculate the median:

```bash
./command_line_stats.sh median
```

---

### Calculate the mode:

```bash
./command_line_stats.sh mode
```

---

## 4. Measures of Dispersion

### Minimum and Maximum:

```bash
./command_line_stats.sh minmax
```

---

### Standard deviation (population):

```bash
./command_line_stats.sh stdpop
```

---

### Standard deviation (sample):

```bash
./command_line_stats.sh stdsample
```

---

### Variance:

```bash
./command_line_stats.sh variance
```

---

## 5. Percentiles

### Quartiles (Q1, Median, Q3):

```bash
./command_line_stats.sh quartiles
```

---

### Custom percentile (say, the 90th percentile):

```bash
./command_line_stats.sh percentile 90
```

---

## 6. Multi-Column Statistics

### Calculate averages for all numeric columns:

```bash
./command_line_stats.sh averages
```

---

## 7. Correlation

### Pearson correlation (visitors vs page_views):

```bash
./command_line_stats.sh correlation
```

---

## 8. Help Menu

Display all available commands:

```bash
./command_line_stats.sh
```

or

```bash
./command_line_stats.sh help
```



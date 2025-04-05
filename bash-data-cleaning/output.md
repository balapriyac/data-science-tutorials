```
$ head users.csv
id,first_name,last_name,email,signup_date,last_login,purchase_amount
1,John,Smith,john.smith@example.com,2023-01-15,2023-03-20,125.99
2,Jane,Doe,jane.doe@example.com,2023-01-16,2023-03-21,210.50
3,Bob,Johnson,bob@example.com,2023-01-17,2023-03-22,0
4,Alice,Williams,alice.williams@example.com,2023-01-18,,75.25
5,,Brown,mike.brown@example.com,2023-01-19,2023-03-24,150.75
6,Sarah,Miller,sarah.miller@example.com,invalid_date,2023-03-25,95.00
7,David,Jones,david.jones@example.com,2023-01-21,2023-03-26,300.00
8,Lisa,Garcia,lisa.garcia@example.com,2023-01-22,2023-03-27,-50.00
9,James,Martinez,mymail@example.com,2023-01-23,2023-03-28,125.00
```

```
$ head -n 5 users.csv
id,first_name,last_name,email,signup_date,last_login,purchase_amount
1,John,Smith,john.smith@example.com,2023-01-15,2023-03-20,125.99
2,Jane,Doe,jane.doe@example.com,2023-01-16,2023-03-21,210.50
3,Bob,Johnson,bob@example.com,2023-01-17,2023-03-22,0
4,Alice,Williams,alice.williams@example.com,2023-01-18,,75.25
```
```
$ grep -c ",," users.csv
2
```

```
grep -n ",," users.csv
5:4,Alice,Williams,alice.williams@example.com,2023-01-18,,75.25
6:5,,Brown,mike.brown@example.com,2023-01-19,2023-03-24,150.75
```

```
$ grep -v -E '^[0-9]{4}-[0-9]{2}-[0-9]{2}$' users.csv | grep "invalid_date"
6,Sarah,Miller,sarah.miller@example.com,invalid_date,2023-03-25,95.00

```

```
$ awk -F, '$7 < 0 {print $0}' users.csv
8,Lisa,Garcia,lisa.garcia@example.com,2023-01-22,2023-03-27,-50.00
```

```
$ awk -F, 'NR>1 {sum += $7} END {print "Total purchases: $" sum}' users_cleaned.csv
```
```
Total purchases: $1282.49
```

```
balapriya@balapriya-82C4:~/bash-data-cleaning$ awk -F, 'NR>1 {sum += $7; count++} END {print "Average purchase: $" sum/count}' users_cleaned.csv
Average purchase: $128.249
```
```
$ awk -F, 'NR>1 {
    split($5, date, "-");
    months[date[2]]++;
} 
END {
    for (month in months) {
        print "Month " month ": " months[month] " users"
    }
}' users_cleaned.csv
Month 01: 10 users
```

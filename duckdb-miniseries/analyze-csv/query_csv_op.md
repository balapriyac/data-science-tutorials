```
┌─────────────┬───────────────┬───────┬───┬───────────────┬────────────────┐
│ customer_id │ customer_name │  age  │ … │ purchase_date │ payment_method │
│    int64    │    varchar    │ int64 │   │     date      │    varchar     │
├─────────────┼───────────────┼───────┼───┼───────────────┼────────────────┤
│           1 │ Customer 1    │    56 │ … │ 2024-01-01    │ PayPal         │
│           2 │ Customer 2    │    46 │ … │ 2024-01-02    │ Credit Card    │
│           3 │ Customer 3    │    32 │ … │ 2024-01-03    │ Cash           │
│           4 │ Customer 4    │    60 │ … │ 2024-01-04    │ Cash           │
│           5 │ Customer 5    │    25 │ … │ 2024-01-05    │ Debit Card     │
├─────────────┴───────────────┴───────┴───┴───────────────┴────────────────┤
│ 5 rows                                               8 columns (5 shown) │
└──────────────────────────────────────────────────────────────────────────┘
```

```
┌─────────────────┬─────────────┬─────────┬─────────┬─────────┬─────────┐
│   column_name   │ column_type │  null   │   key   │ default │  extra  │
│     varchar     │   varchar   │ varchar │ varchar │ varchar │ varchar │
├─────────────────┼─────────────┼─────────┼─────────┼─────────┼─────────┤
│ customer_id     │ BIGINT      │ YES     │ NULL    │ NULL    │ NULL    │
│ customer_name   │ VARCHAR     │ YES     │ NULL    │ NULL    │ NULL    │
│ age             │ BIGINT      │ YES     │ NULL    │ NULL    │ NULL    │
│ gender          │ VARCHAR     │ YES     │ NULL    │ NULL    │ NULL    │
│ purchase_amount │ DOUBLE      │ YES     │ NULL    │ NULL    │ NULL    │
│ category        │ VARCHAR     │ YES     │ NULL    │ NULL    │ NULL    │
│ purchase_date   │ DATE        │ YES     │ NULL    │ NULL    │ NULL    │
│ payment_method  │ VARCHAR     │ YES     │ NULL    │ NULL    │ NULL    │
└─────────────────┴─────────────┴─────────┴─────────┴─────────┴─────────┘
```

```
┌─────────┬─────────┬───┬──────────────┬────────────────────┐
│ min_age │ max_age │ … │ max_purchase │    avg_purchase    │
│  int64  │  int64  │   │    double    │       double       │
├─────────┼─────────┼───┼──────────────┼────────────────────┤
│   19    │   61    │ … │    485.1     │ 242.92866666666666 │
├─────────┴─────────┴───┴──────────────┴────────────────────┤
│ 1 rows                                6 columns (4 shown) │
└───────────────────────────────────────────────────────────┘
```

```
┌───────────────┬───────┬─────────────────┬─────────────┐
│ customer_name │  age  │ purchase_amount │  category   │
│    varchar    │ int64 │     double      │   varchar   │
├───────────────┼───────┼─────────────────┼─────────────┤
│ Customer 16   │    20 │           485.1 │ Groceries   │
│ Customer 18   │    19 │          470.35 │ Groceries   │
│ Customer 21   │    47 │          461.72 │ Groceries   │
│ Customer 9    │    40 │          455.57 │ Groceries   │
│ Customer 19   │    41 │          448.47 │ Electronics │
│ Customer 28   │    61 │          416.08 │ Clothing    │
│ Customer 1    │    56 │          406.11 │ Books       │
│ Customer 17   │    39 │          389.82 │ Groceries   │
│ Customer 4    │    60 │          345.27 │ Electronics │
│ Customer 11   │    28 │          334.64 │ Books       │
│ Customer 20   │    61 │          302.97 │ Books       │
│ Customer 14   │    57 │          277.89 │ Books       │
│ Customer 13   │    53 │          264.83 │ Books       │
│ Customer 7    │    56 │          252.64 │ Electronics │
│ Customer 5    │    25 │          225.67 │ Groceries   │
│ Customer 26   │    29 │          200.45 │ Clothing    │
├───────────────┴───────┴─────────────────┴─────────────┤
│ 16 rows                                     4 columns │
└───────────────────────────────────────────────────────┘
```

```
┌─────────────┬─────────────────┬────────────────────┬────────────────────┐
│  category   │ total_purchases │    total_sales     │     avg_spent      │
│   varchar   │      int64      │       double       │       double       │
├─────────────┼─────────────────┼────────────────────┼────────────────────┤
│ Groceries   │               9 │ 2716.7100000000005 │ 301.85666666666674 │
│ Electronics │              12 │            2119.94 │ 176.66166666666666 │
│ Books       │               7 │ 1834.6799999999998 │ 262.09714285714284 │
│ Clothing    │               2 │             616.53 │            308.265 │
└─────────────┴─────────────────┴────────────────────┴────────────────────┘
```


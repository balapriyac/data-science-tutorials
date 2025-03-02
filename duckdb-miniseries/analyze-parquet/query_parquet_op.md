```
┌──────────┬───────────────┬───┬─────────────────────┬────────────────┐
│ order_id │ customer_name │ … │     order_time      │ payment_method │
│  int64   │    varchar    │   │       varchar       │    varchar     │
├──────────┼───────────────┼───┼─────────────────────┼────────────────┤
│        1 │ Grace         │ … │ 2024-02-01 18:00:00 │ PayPal         │
│        2 │ David         │ … │ 2024-02-01 18:05:00 │ Credit Card    │
│        3 │ Eve           │ … │ 2024-02-01 18:10:00 │ PayPal         │
│        4 │ Grace         │ … │ 2024-02-01 18:15:00 │ PayPal         │
│        5 │ Charlie       │ … │ 2024-02-01 18:20:00 │ Debit Card     │
├──────────┴───────────────┴───┴─────────────────────┴────────────────┤
│ 5 rows                                          8 columns (4 shown) │
└─────────────────────────────────────────────────────────────────────┘
```

```
┌────────────────┬─────────────┬─────────┬─────────┬─────────┬─────────┐
│  column_name   │ column_type │  null   │   key   │ default │  extra  │
│    varchar     │   varchar   │ varchar │ varchar │ varchar │ varchar │
├────────────────┼─────────────┼─────────┼─────────┼─────────┼─────────┤
│ order_id       │ BIGINT      │ YES     │ NULL    │ NULL    │ NULL    │
│ customer_name  │ VARCHAR     │ YES     │ NULL    │ NULL    │ NULL    │
│ table_number   │ BIGINT      │ YES     │ NULL    │ NULL    │ NULL    │
│ menu_item      │ VARCHAR     │ YES     │ NULL    │ NULL    │ NULL    │
│ price          │ DOUBLE      │ YES     │ NULL    │ NULL    │ NULL    │
│ quantity       │ BIGINT      │ YES     │ NULL    │ NULL    │ NULL    │
│ order_time     │ VARCHAR     │ YES     │ NULL    │ NULL    │ NULL    │
│ payment_method │ VARCHAR     │ YES     │ NULL    │ NULL    │ NULL    │
└────────────────┴─────────────┴─────────┴─────────┴─────────┴─────────┘


```

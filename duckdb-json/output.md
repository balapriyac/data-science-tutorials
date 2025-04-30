```
┌──────────┬───┬──────────────────────┬──────────────────────┐
│ order_id │ … │        items         │       payment        │
│ varchar  │   │ struct(product_id …  │ struct("method" va…  │
├──────────┼───┼──────────────────────┼──────────────────────┤
│ ORD-1001 │ … │ [{'product_id': PR…  │ {'method': credit_…  │
│ ORD-1002 │ … │ [{'product_id': PR…  │ {'method': paypal,…  │
├──────────┴───┴──────────────────────┴──────────────────────┤
│ 2 rows                                 5 columns (3 shown) │
└────────────────────────────────────────────────────────────┘
```

```
┌─────────────┐
│ order_count │
│    int64    │
├─────────────┤
│      2      │
└─────────────┘
```

```
┌──────────┬───────────────┐
│ order_id │ customer_name │
│ varchar  │    varchar    │
├──────────┼───────────────┤
│ ORD-1001 │ Alex Johnson  │
│ ORD-1002 │ Sarah Miller  │
└──────────┴───────────────┘
```

```
┌──────────┬───────────────┬─────────┬─────────┐
│ order_id │ customer_name │  city   │  state  │
│ varchar  │    varchar    │ varchar │ varchar │
├──────────┼───────────────┼─────────┼─────────┤
│ ORD-1001 │ Alex Johnson  │ Boston  │ MA      │
│ ORD-1002 │ Sarah Miller  │ Seattle │ WA      │
└──────────┴───────────────┴─────────┴─────────┘
```

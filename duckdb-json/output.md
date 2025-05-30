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
```
┌──────────┬───────────────┐
│ order_id │ customer_name │
│ varchar  │    varchar    │
├──────────┼───────────────┤
│ ORD-1002 │ Sarah Miller  │
└──────────┴───────────────┘
```

```
┌──────────┬────────────────┬───────────────┐
│ order_id │ payment_method │ total_amount  │
│ varchar  │    varchar     │ decimal(18,3) │
├──────────┼────────────────┼───────────────┤
│ ORD-1001 │ credit_card    │       179.970 │
│ ORD-1002 │ paypal         │       137.960 │
└──────────┴────────────────┴───────────────┘
```

```
┌──────────┬───────────────┬───┬───────────────┬──────────┐
│ order_id │ customer_name │ … │     price     │ quantity │
│ varchar  │    varchar    │   │ decimal(18,3) │  int32   │
├──────────┼───────────────┼───┼───────────────┼──────────┤
│ ORD-1001 │ Alex Johnson  │ … │       129.990 │        1 │
│ ORD-1001 │ Alex Johnson  │ … │        24.990 │        2 │
│ ORD-1002 │ Sarah Miller  │ … │        89.990 │        1 │
│ ORD-1002 │ Sarah Miller  │ … │        15.990 │        3 │
├──────────┴───────────────┴───┴───────────────┴──────────┤
│ 4 rows                              6 columns (4 shown) │
└─────────────────────────────────────────────────────────┘
```

```
┌──────────┬───────────────┬───────────────┬────────────┐
│ order_id │ customer_name │  order_total  │ item_count │
│ varchar  │    varchar    │ decimal(18,3) │   uint64   │
├──────────┼───────────────┼───────────────┼────────────┤
│ ORD-1001 │ Alex Johnson  │       179.970 │          2 │
│ ORD-1002 │ Sarah Miller  │       137.960 │          2 │
└──────────┴───────────────┴───────────────┴────────────┘
```
```
┌─────────────────┬───────────┐
│    category     │ avg_price │
│     varchar     │  double   │
├─────────────────┼───────────┤
│ Electronics     │    129.99 │
│ Kitchen         │     89.99 │
│ Accessories     │     24.99 │
│ Food & Beverage │     15.99 │
└─────────────────┴───────────┘
```
```
┌──────────┬───────────────┬───┬───────────────┬──────────┐
│ order_id │ customer_name │ … │     price     │ quantity │
│ varchar  │    varchar    │   │ decimal(18,3) │  int32   │
├──────────┼───────────────┼───┼───────────────┼──────────┤
│ ORD-1001 │ Alex Johnson  │ … │       129.990 │        1 │
│ ORD-1001 │ Alex Johnson  │ … │        24.990 │        2 │
│ ORD-1002 │ Sarah Miller  │ … │        89.990 │        1 │
│ ORD-1002 │ Sarah Miller  │ … │        15.990 │        3 │
├──────────┴───────────────┴───┴───────────────┴──────────┤
│ 4 rows                              6 columns (4 shown) │
└─────────────────────────────────────────────────────────┘
```

```
┌──────────┬───────────────┬───────────────────────────────────────────────────┐
│ order_id │ customer_name │                       item                        │
│ varchar  │    varchar    │ struct(product_id varchar, "name" varchar, cate…  │
├──────────┼───────────────┼───────────────────────────────────────────────────┤
│ ORD-1001 │ Alex Johnson  │ {'product_id': PROD-501, 'name': Wireless Headp…  │
│ ORD-1001 │ Alex Johnson  │ {'product_id': PROD-245, 'name': Smartphone Cas…  │
│ ORD-1002 │ Sarah Miller  │ {'product_id': PROD-103, 'name': Coffee Maker, …  │
│ ORD-1002 │ Sarah Miller  │ {'product_id': PROD-107, 'name': Coffee Beans P…  │
└──────────┴───────────────┴───────────────────────────────────────────────────┘
```

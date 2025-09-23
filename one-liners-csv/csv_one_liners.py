path = "/content/data.csv"

print(f"Total: ${sum(float(r[3]) for r in __import__('csv').reader(open(path)) if r[0] != 'transaction_id'):,.2f}")

print(max({r[5]:sum(float(row[3]) for row in __import__('csv').reader(open(path)) if row[5]==r[5] and row[0]!='transaction_id') for r in __import__('csv').reader(open(path)) if r[0]!='transaction_id'}.items(), key=lambda x:x[1]))

[print(f"{r[1]}: ${float(r[3]):,.2f}") for r in __import__('csv').reader(open(path)) if r[7]=='Enterprise' and r[0]!='transaction_id']

print({r:f"${sum(float(row[3]) for row in __import__('csv').reader(open(path)) if row[6]==r and row[0]!='transaction_id'):,.2f}" for r in set(row[6] for row in __import__('csv').reader(open(path)) if row[0]!='transaction_id')})

print(sorted([(r[1], f"${float(r[3]):,.2f}") for r in list(__import__('csv').reader(open(path)))[1:] if float(r[3]) > 100000], key=lambda x: float(x[1][1:].replace(',','')), reverse=True))

print(len(set(r[2] for r in __import__('csv').reader(open(path)) if r[0] != 'transaction_id')))

print(f"Average: ${sum(float(r[3]) for r in __import__('csv').reader(open(path)) if r[6]=='North America' and r[0]!='transaction_id') / len([r for r in __import__('csv').reader(open('data.csv')) if r[6]=='North America' and r[0]!='transaction_id']):,.2f}")

[print(f"{r[1]} | {r[2]} | ${r[3]}") for r in __import__('csv').reader(open(path)) if r[2]=='Software' and float(r[3]) > 50000 and r[0] != 'transaction_id']

vals=[float(r[3]) for r in __import__('csv').reader(open(path)) if r[0]!='transaction_id']; print(f"Min: ${min(vals):,.2f} | Max: ${max(vals):,.2f} | Avg: ${sum(vals)/len(vals):,.2f}")
print(vals)

__import__('csv').writer(open('filtered.csv','w',newline='')).writerows([r for r in list(__import__('csv').reader(open(path)))[1:] if float(r[3]) > 75000])

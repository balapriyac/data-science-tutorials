import re

text = "Data science is cool as you get to work with real-world data"
matches = re.findall(r"data", text)
print(matches)

matches = re.findall(r"data", text, re.IGNORECASE)
print(matches)

text = "The cat sat on the mat. The bat flew over the rat."
pattern = r"The ... "
matches = re.findall(pattern, text)
print(matches)

text = "The cat sat on the mat. The bat flew over the rat."
pattern = r"[cb]at"
matches = re.findall(pattern, text)
print(matches)
     

# Find all lowercase words that start with a-d
pattern = r"\b[a-d][a-z]*\b"
text = "apple banana cherry date elephant fig grape kiwi lemon mango orange"
matches = re.findall(pattern, text)
print(matches)


text = "Contact: john.doe@example.com"
pattern = r"(?P[\w.]+)@(?P[\w.]+)"

match = re.search(pattern, text)
if match:
    print(f"Username: {match.group('username')}")
    print(f"Domain: {match.group('domain')}")

     
     
text = "Phone numbers: 555-1234, 555-5678, 5551234"
pattern = r"\b\d{3}-?\d{4}\b"
matches = re.findall(pattern, text)
print(matches)



text = "Python is popular in data science."

# ^ anchors to the start of the string
start_matches = re.findall(r"^Python", text)
print(start_matches)

# $ anchors to the end of the string
end_matches = re.findall(r"science\.$", text)
print(end_matches)
     
text = "Dates: 2023-10-15, 2022-05-22"
pattern = r"(\d{4})-(\d{2})-(\d{2})"

# findall returns tuples of the captured groups
matches = re.findall(pattern, text)
print(matches)

# You can use these to create structured data
for year, month, day in matches:
    print(f"Year: {year}, Month: {month}, Day: {day}")


text = "Contact: john.doe@example.com"
pattern = r"(?P[\w.]+)@(?P[\w.]+)"

match = re.search(pattern, text)
if match:
    print(f"Username: {match.group('username')}")
    print(f"Domain: {match.group('domain')}")

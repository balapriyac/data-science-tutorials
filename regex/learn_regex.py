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
     
text = "Phone numbers: 555-1234, 555-5678, 5551234"
pattern = r"\b\d{3}-?\d{4}\b"
matches = re.findall(pattern, text)
print(matches)

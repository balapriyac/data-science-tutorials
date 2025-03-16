# Regular Expressions Quick Reference Table

## Basic Metacharacters

| Character | Description | Example | Matches |
|-----------|-------------|---------|---------|
| `.` | Any character except newline | `a.b` | "acb", "adb", "a3b", etc. |
| `^` | Start of string | `^Hello` | "Hello world" but not "Say Hello" |
| `$` | End of string | `world$` | "Hello world" but not "world class" |
| `*` | 0 or more repetitions | `ab*c` | "ac", "abc", "abbc", etc. |
| `+` | 1 or more repetitions | `ab+c` | "abc", "abbc", etc. but not "ac" |
| `?` | 0 or 1 repetition | `ab?c` | "ac", "abc" but not "abbc" |
| `{n}` | Exactly n repetitions | `a{3}` | "aaa" |
| `{m,n}` | m to n repetitions | `a{2,4}` | "aa", "aaa", "aaaa" |
| `{m,}` | m or more repetitions | `a{2,}` | "aa", "aaa", "aaaa", etc. |
| `\` | Escape character | `\.` | Literal period "." |
| `[]` | Character class | `[abc]` | "a", "b", or "c" |
| `\|` | Alternation (OR) | `cat\|dog` | "cat" or "dog" |
| `()` | Grouping | `(ab)+` | "ab", "abab", etc. |

## Character Classes

| Expression | Description | Equivalent | 
|------------|-------------|------------|
| `\d` | Any digit | `[0-9]` |
| `\D` | Any non-digit | `[^0-9]` |
| `\w` | Any word character | `[a-zA-Z0-9_]` |
| `\W` | Any non-word character | `[^a-zA-Z0-9_]` |
| `\s` | Any whitespace | `[ \t\n\r\f\v]` |
| `\S` | Any non-whitespace | `[^ \t\n\r\f\v]` |
| `[abc]` | Any of listed characters | - |
| `[^abc]` | Any character except listed | - |
| `[a-z]` | Any character in range | - |

## Assertions

| Expression | Description |
|------------|-------------|
| `(?=...)` | Positive lookahead |
| `(?!...)` | Negative lookahead |
| `(?<=...)` | Positive lookbehind |
| `(?<!...)` | Negative lookbehind |
| `\b` | Word boundary |
| `\B` | Not a word boundary |

## Common Python Regex Functions

| Function | Description |
|----------|-------------|
| `re.search(pattern, string)` | Returns first match or None |
| `re.match(pattern, string)` | Matches at start of string |
| `re.findall(pattern, string)` | Returns all matches as list |
| `re.finditer(pattern, string)` | Returns iterator of match objects |
| `re.sub(pattern, repl, string)` | Substitutes matches with replacement |
| `re.split(pattern, string)` | Splits string by pattern |
| `re.compile(pattern)` | Compiles pattern for reuse |

## Common Patterns for Data Science

| Task | Pattern | Example |
|------|---------|---------|
| Email | `[\w.%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}` | user@example.com |
| Phone (US) | `\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}` | (555) 123-4567 |
| Date (ISO) | `\d{4}-\d{2}-\d{2}` | 2023-10-15 |
| URL | `https?://[^\s/$.?#].[^\s]*` | https://example.com |
| IP Address | `\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b` | 192.168.1.1 |
| Credit Card | `\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b` | 1234-5678-9012-3456 |
| ZIP Code (US) | `\b\d{5}(?:-\d{4})?\b` | 12345 or 12345-6789 |

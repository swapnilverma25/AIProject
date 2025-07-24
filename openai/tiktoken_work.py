import tiktoken
encoding = tiktoken.get_encoding("cl100k_base")
tokens = encoding.encode("this is my git repo")
print(tokens) # ID of tokens generated
print(len(tokens)) # Count of tokens

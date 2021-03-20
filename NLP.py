import requests

url = "https://api.promptapi.com/nlp/lemma?lang=lang"

payload = "Apply some text here to apply algorithm 3aleeeeeh to check it's working".encode("utf-8")
headers= {
  "apikey": "DmCJVSkpUWGbl81MFgI8hf59HgULrGMS"
}

response = requests.request("POST", url, headers=headers, data = payload)

status_code = response.status_code
result = response.text
print(result)
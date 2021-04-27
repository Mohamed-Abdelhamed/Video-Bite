import requests

url = "https://api.promptapi.com/keyword"

 class Keywords:
  def __init__(self,payload): 
    self.payload = payload
  def extract_keywords(self):
    
    headers = {"apikey": "DmCJVSkpUWGbl81MFgI8hf59HgULrGMS"}
    response = requests.request("POST", url, headers=headers, data = payload)
    status_code = response.status_code
    result = response.text
    print(result)
    
payload = "Nothing prepared them for moving into the former country estate of the Guinness family," \
          " with a 32-seat cinema, two gourmet dining rooms, a spa, wine-tasting tunnels and 83 guest rooms." \
          "It was transformed into a luxury hotel in 1939; past guests have included US presidents and " \
          "celebrities such as Barbra Streisand and Brad Pitt. Pierce Brosnan, who shot an episode of the " \
          "TV series Remington Steele at Ashford, returned to marry Keely Shaye Smith here in 2001.Both Smith " \
          "and Jamieson, who is in charge of guest services, were surprised -- and " \
          "thrilled -- when general manager Niall Rochford asked them if they might consider moving " \
          "in for a spell. They suspect it's because they complement each other as a team.She does everything " \
          "so well on the inside and I have a lot of experience on the outside and on the grounds, " \
          "so it was almost perfect yin and yang, says Smith. We balance each other. So I have to believe " \
          "that's one of the reasons they asked.".encode("utf-8")
x=Keywords(payload)


x.extract_keywords()

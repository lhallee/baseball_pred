import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "https://www.baseball-reference.com/leagues/NL/2023-standard-fielding.shtml"

response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

table = soup.find('table', {'id': 'players_standard_fielding'})

# Get table headers
headers = [th.text for th in table.find('thead').findAll('th')]

# Get table rows
rows = table.find('tbody').findAll('tr')

# Get table data
data = []
for row in rows:
    cols = row.findAll('td')
    cols = [col.text for col in cols]
    data.append(cols)

# Create DataFrame
df = pd.DataFrame(data, columns=headers)
print(df)
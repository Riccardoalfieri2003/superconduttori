import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
import pandas as pd

def createDriver():
    #cambiare il chromedriver quando necessario
    service = Service(executable_path=r'C:\Users\rical\OneDrive\Desktop\chromedriver-win64\chromedriver.exe')
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument("--log-level=1")
    options.add_argument('--log-level=3')
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--ignore-ssl-errors')
    options.page_load_strategy = 'eager'

    driver = webdriver.Chrome(service=service, options=options) 
    return driver

driver=createDriver()

data = []
numbers = [f"{i:03}" for i in range(1, 119)]

for number in numbers:

    driver.get("https://periodictable.com/Elements/"+number+"/data.html")

    keys = driver.find_elements(By.XPATH, '//td[@align="right"]')
    values = driver.find_elements(By.XPATH, '//td[@align="left" and not(@colspan="2")]')

    cleaned_values = []
    for value in values:
        clean_text = value.text.replace('[note]','').replace('\n', ' ')
        cleaned_values.append(clean_text)

     # Ensure keys and values match
    if len(keys) != len(cleaned_values):
        print(f"Mismatch on page {number}: {len(keys)} keys, {len(cleaned_values)} values")
        continue
    
    # Create a dictionary for the current row
    row = {key.text: cleaned_values for key, cleaned_values in zip(keys, cleaned_values)}
    print(row)
    data.append(row)


    print("Element n."+str(number)+" scraped")

# Close the WebDriver
driver.quit()

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Save the dataset to a CSV file
df.to_csv("elements_data.csv", index=False)

print("Scraping completed. Data saved.")
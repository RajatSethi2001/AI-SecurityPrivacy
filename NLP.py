import re
from selenium import webdriver
from selenium.webdriver.firefox.options import Options

options = Options()
options.add_argument("--headless")
profile = webdriver.FirefoxProfile() 
profile.add_extension(extension='adblock_plus-3.11.4-an fx.xpi')
options.binary_location = 'C:\\Program Files\\Mozilla Firefox\\firefox.exe'
driver = webdriver.Firefox(options=options, firefox_profile=profile, executable_path="C:\\Users\\Admin\\Desktop\\AI-SEC\\geckodriver.exe")

ratingFile = open("ratings.txt", "w+")

def main():
    reviewStart = 5000000
    numReviews = 10000
    for i in range(reviewStart, reviewStart + numReviews):
        #print(reviewStart+i)
        url = f"https://www.imdb.com/review/rw{i}"
        #print(url)
        driver.get(url)
        page_source = driver.page_source
        #print(page_source)
        try:
            reviewStr = re.search("reviewBody\": \"(.*)\",\n  \"reviewRating\"", page_source).group(1)
            ratingStr = re.search("ratingValue\": \"([0-9]*)\"", page_source).group(1)
            ratingLine = f"{ratingStr}||{reviewStr}\n"
            ratingFile.write(ratingLine)
            print(ratingLine)
        except:
            pass

main()
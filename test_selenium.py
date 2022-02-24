from selenium import webdriver 
import time # 크롬창(웹드라이버) 열기 
driver = webdriver.Chrome("./chromedriver") # 구글 지도 접속하기 
driver.get("https://www.google.com/maps/") # 검색창에 "카페" 입력하기 
searchbox = driver.find_element_by_css_selector("input#searchboxinput") 
searchbox.send_keys("카페") # 검색버튼 누르기 
searchbutton = driver.find_element_by_css_selector("button#searchbox-searchbutton") 
searchbutton.click()
for i in range(999):
    time.sleep(3)

    stores = driver.find_elements_by_css_selector("div.section-result-content") 

    for s in stores:
        title = s.find_element_by_css_selector("h3.section-result-title").text
          
        try: 
            score = s.find_element_by_css_selector("span.cards-rating-score").text 
        except: score = "평점없음" 
        addr = s.find_element_by_css_selector("span.section-result-location").text 
        print(title, "/", score, "/", addr)
    try: 
        nextpage = driver.find_element_by_css_selector("button#n7lv7yjyC35__section-pagination-button-next") 
        nextpage.click() 
    except: 
        print("데이터 수집 완료.") 
        break
    
driver.close()
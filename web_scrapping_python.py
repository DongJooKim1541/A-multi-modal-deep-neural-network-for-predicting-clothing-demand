import time
import urllib.request
from bs4 import BeautifulSoup
from selenium import webdriver
import os
from time import sleep


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def createCsv():
    f = open("info.csv", "a")
    f.write("조회수, 누적 판매량, 성별, 선호 성별, 선호 연령대, 가격, 카테고리1, 카테고리2\n")
    f.close()


def writeCsv(views, cumulative_sales, gender, best_gender, best_age, price, category1, category2):
    f = open("info.csv", "a")

    views = views.strip()
    views = views.replace("회 이상", "")
    if "천" in views:
        views = views.replace("천", "")
        views = str(float(views) * 1000)
    elif "만" in views:
        views = views.replace("만", "")
        views = str(float(views) * 10000)
    elif "십만" in views:
        views = views.replace("십만", "")
        views = str(float(views) * 100000)

    cumulative_sales = cumulative_sales.strip()
    cumulative_sales = cumulative_sales.replace("개 이상", "")

    gender = gender.strip()
    if gender == "남":
        gender = "0"
    elif gender == "여":
        gender = "1"
    else:
        gender = "2"

    best_gender = best_gender.strip()
    if best_gender == "남":
        best_gender = "0"
    elif best_gender == "여":
        best_gender = "1"
    else:
        best_gender = "2"

    best_age = best_age.strip()
    price = price.strip()
    price = price.replace(",", "")
    price = price.replace("원", "")
    category1 = category1.strip()
    category2 = category2.strip()

    print("views: ", views)
    print("cumulative_sales: ", cumulative_sales)
    print("gender: ", gender)
    print("best_gender: ", best_gender)
    print("best_age: ", best_age)
    print("price: ", price)
    print("category1: ", category1)
    print("category2: ", category2)

    f.write(
        views + "," + cumulative_sales + "," + gender + "," + best_gender + "," + best_age + "," + price + "," + category1 + "," + category2 + "\n")
    f.close()


# 스크립트를 실행하려면 여백의 녹색 버튼을 누릅니다.
if __name__ == '__main__':
    start_time = time.time()
    #createCsv()
    for i in range(800000, 9999999 + 1):
        delay = 2
        # 웹 사이트 파싱
        try:
            chrome_driver_path = "C:/Users/vml/kdj/chromedriver.exe"
            browser = webdriver.Chrome(chrome_driver_path)
            url = "https://www.musinsa.com/app/goods/" + str(i)
            browser.get(url)
            bsObject = BeautifulSoup(browser.page_source, "html.parser")
        except:
            # 웹 사이트가 정상적으로 열리지 않을 경우
            print(i,"번째 데이터 수집 실패!")
            continue
        sleep(3)
        # 성별 추출
        gender = ""
        made_for_gender1 = bsObject.select_one(
            '#product_order_info > div.explan_product.product_info_section > ul > li:nth-child(2) > p.product_article_contents > span.txt_gender > span:nth-child(1)')
        made_for_gender2 = bsObject.select_one(
            '#product_order_info > div.explan_product.product_info_section > ul > li:nth-child(2) > p.product_article_contents > span.txt_gender > span:nth-child(2)')
        if made_for_gender2 is not None:
            gender = made_for_gender1.get_text() + made_for_gender2.get_text()
        else:
            gender = bsObject.select_one(
                '#product_order_info > div.explan_product.product_info_section > ul > li:nth-child(2) > p.product_article_contents > span.txt_gender')
            if gender is not None:
                gender=gender.get_text()

        # 조회수 추출
        views = bsObject.select_one('#pageview_1m').get_text()

        # 누적 판매량 추출
        cumulative_sales = bsObject.select_one('#sales_1y_qty').get_text()

        # 선호 연령대 추출
        best_age = bsObject.select_one('#graph_summary_area > strong:nth-child(2) > em')

        # 조회수와 누적 판매량과 선호 연령대는 상품에 따라 없을 수 있기 때문에 셋 중 하나라도 0이면 웹 스크래핑 종료
        if len(views) == 0 or len(cumulative_sales) == 0 or len(best_age) == 0:
            print(i,"번째 데이터 수집 실패!")
            browser.quit()
            continue
        else:
            # 가격 추출
            price = bsObject.select_one('#sPrice > ul > li.pertinent > span.txt_price_member').get_text()
            # 선호 성별
            best_gender = bsObject.select_one('#graph_summary_area > span.man.graph_sex_text')

            if made_for_gender1 and made_for_gender2 and best_gender is not None:
                best_gender = best_gender.get_text()
            else:
                best_gender = gender

            best_age = best_age.get_text()

            # 카테고리 추출
            category1 = bsObject.select_one(
                '#page_product_detail > div.right_area.page_detail_product > div.right_contents.section_product_summary > div.product_info > p > a:nth-child(1)').get_text()
            category2 = bsObject.select_one(
                '#page_product_detail > div.right_area.page_detail_product > div.right_contents.section_product_summary > div.product_info > p > a:nth-child(2)')
            if category2 is not None:
                category2 = category2.get_text()
                folder_path = "./images/" + category1.strip() + "/" + category2.strip()
                createFolder(folder_path)
                writeCsv(views, cumulative_sales, gender, best_gender, best_age, price, category1, category2)
            else:
                folder_path = "./images/" + category1.strip()
                createFolder(folder_path)
                writeCsv(views, cumulative_sales, gender, best_gender, best_age, price, category1, category2="None")

            # 의류 썸네일 추출 및 저장
            clothes_thumbnail = bsObject.find_all(class_='product-img')
            if len(clothes_thumbnail) != 0:
                product_img_url = "https:" + bsObject.select_one('#bigimg')['src']

                urllib.request.urlretrieve(product_img_url, folder_path + "/image" + str(i) + ".png")
            print(i, "번째 수집 성공!")
            browser.quit()
    end_time = time.time() - start_time
    print("웹 스크래핑 하는데 걸리는 시간: ", end_time / 60, "min")

import requests
import csv
import time
from urllib.parse import quote
from bs4 import BeautifulSoup


def crawl_taobao_comments(keyword, pages=5):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.190 Safari/537.36',
        'Referer': 'https://www.taobao.com/list/product/iphone13.htm',
    }

    url_template = 'https://s.taobao.com/search?q={}&s={}'
    keyword_encoded = quote(keyword)

    with open('iphone_comments.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Comment']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for page in range(pages):
            url = url_template.format(keyword_encoded, page * 44)
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')

            items = soup.find_all('div', class_='J_MouserOnverReq')
            if not items:
                print(f"No items found on page {page + 1}")
                continue

            for item in items:
                comment = item.find('p', class_='comment')
                if comment:
                    comment_text = comment.text.strip()
                    writer.writerow({'Comment': comment_text})
                    print(f'Comment: {comment_text}')
                else:
                    print("No comment found")

            # 随机延时，模拟人的行为
            time.sleep(1 + 2 * random.random())  # 随机延时范围为 1 到 3 秒

            print(f'第 {page + 1} 页评论爬取完成')


if __name__ == "__main__":
    import random

    keyword = 'iPhone13'  # 搜索关键词
    pages = 5  # 爬取的页数
    crawl_taobao_comments(keyword, pages)

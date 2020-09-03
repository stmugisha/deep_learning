#Abstract scraper

from urllib import request

from bs4 import BeautifulSoup


# some sites close their content for 'bots', so user-agent must be supplied
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
}


class AbstractScraper():

    def __init__(self, url, test=False):
        if test:  # when testing, we load a file
            with url:
                self.soup = BeautifulSoup(url.read(), "html.parser")
        else:
            self.soup = BeautifulSoup(request.urlopen(
                request.Request(url, headers=HEADERS)).read(), "html.parser")

    def host(self):
        """ get the host of the url, so we can use the correct scraper (check __init__.py) """
        raise NotImplementedError("This should be implemented.")

    def title(self):
        raise NotImplementedError("This should be implemented.")

    def total_time(self):
        """ total time it takes to preparate the recipe in minutes """
        raise NotImplementedError("This should be implemented.")

    def ingredients(self):
        raise NotImplementedError("This should be implemented.")

    def instructions(self):
        raise NotImplementedError("This should be implemented.")











































'''
import requests
from bs4 import BeautifulSoup
import pandas as pd

all_url='http://chengdu.8684.cn'
start_html=requests.get(all_url)
Soup=BeautifulSoup(start_html.text,'lxml')
all_a=Soup.find('div',class_='bus_kt_r1').find_all('a')
temp_bus_stop=[]
all_bus_stop=[]

for a in all_a:
    href=a['href']
    html=all_url+href
    start_html2=requests.get(html)
    Soup2=BeautifulSoup(start_html2.text,'lxml')
    all_a2=Soup2.find('div',class_='stie_list').find_all('a')
    for a2 in all_a2:
        temp_bus_stop=[]
        href1=a2['href']
        html2=all_url+href1
        start_html3=requests.get(html2)
        Soup3=BeautifulSoup(start_html3.text,'lxml')
        bus_name=Soup3.find('div',class_='bus_i_t1').find('h1').get_text()
        bus_type=Soup3.find('div',class_='bus_i_t1').find('a').get_text()
        bus_time=Soup3.find_all('p',class_='bus_i_t4')[0].get_text()
        bus_ticket=Soup3.find_all('p',class_='bus_i_t4')[1].get_text()
        bus_company=Soup3.find_all('p',class_='bus_i_t4')[2].find('a').get_text()
        bus_forward_name=Soup3.find('div',class_='bus_line_txt').find('strong').get_text()
        bus_forward_num=Soup3.find('div',class_='bus_line_txt').find('span').get_text()
        bus_forward_stop=Soup3.find('div',class_='bus_site_layer').find_all('div',class_='')
        #all_line = Soup3.find_all('div',class_='bus_line_top')
        #all_site = Soup3.find_all('div',class_='bus_line_site')
        #line_x = all_line[0].find('div',class_='bus_line_txt').get_text()[:-9]+all_line[0].find_all('span')[-1].get_text()
        for stop in bus_forward_stop:
            stop_knot=stop.find('a').get_text()
            temp_bus_stop.append(stop_knot)
        information=[bus_name,bus_type,bus_time,bus_ticket,bus_company,bus_forward_name,bus_forward_num,temp_bus_stop]
        all_bus_stop.append(information)
print(all_bus_stop)

# 定义保存函数，将运算结果保存为txt文件
def text_save(content,filename,mode='a'):
    # Try to save a list variable in txt file.
    file = open(filename,mode)
    for i in range(len(content)):
        file.write(str(content[i])+'\n')
    file.close()

# 输出处理后的数据     
text_save(all_bus_stop,'all_bus_stop.txt')
'''

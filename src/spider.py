import requests
from lxml import etree
import numpy as np
import pandas as pd
import time

# 获取爬取的城市和爬取的页码数
def get_city_page():
    city = input('请输入要爬取的城市：')
    page = int(input('请输入要爬取的页码数：'))
    return city,page

# 构建url_list
def create_url_lis(city,page):
    # 城市对应编码
    city_code_dict = {
        '上海':538, '北京':530, '广州':763, '深圳':765, '天津':531, '武汉':736, '西安':854,
        '成都':801, '南京':635, '杭州':653, '重庆':551, '厦门':682,'大连':600
    }
    url_lis = []
    city_code = city_code_dict[city]
    for p in range(page):
        url = 'https://sou.zhaopin.com/?jl={}&kw={}&p={}'.format(city_code,'数据分析',p+1)
        url_lis.append(url)
    return url_lis

# 根据url获取网页源代码
def get_html(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36',
        'cookie': 'x-zp-client-id=e75d5a0d-3231-46b1-8a83-e63fe8c3de3f; FSSBBIl1UgzbN7NO=5svgccXO9PrSU7PqesTEI.SiP0r5yssganvpazL9DJA9g6YGaua51cMAdik4JAGT7MB3JHzdan4ZO0eQVS2FV5q; locationInfo_search={%22code%22:%22638%22%2C%22name%22:%22%E5%B8%B8%E5%B7%9E%22%2C%22message%22:%22%E5%8C%B9%E9%85%8D%E5%88%B0%E5%B8%82%E7%BA%A7%E7%BC%96%E7%A0%81%22}; _uab_collina=169692390347437664558343; at=6faccf1c1c014d3f8a917e6b95c330b2; rt=feb1c3d5af9741158efb1d5f0afa8ab8; sensorsdata2015jssdkcross=%7B%22distinct_id%22%3A%221169345395%22%2C%22first_id%22%3A%2218b188be2ce11d5-09620ba8a14163-26031e51-1638720-18b188be2cf1f60%22%2C%22props%22%3A%7B%22%24latest_traffic_source_type%22%3A%22%E7%9B%B4%E6%8E%A5%E6%B5%81%E9%87%8F%22%2C%22%24latest_search_keyword%22%3A%22%E6%9C%AA%E5%8F%96%E5%88%B0%E5%80%BC_%E7%9B%B4%E6%8E%A5%E6%89%93%E5%BC%80%22%2C%22%24latest_referrer%22%3A%22%22%7D%2C%22identities%22%3A%22eyIkaWRlbnRpdHlfY29va2llX2lkIjoiMThiMTg4YmUyY2UxMWQ1LTA5NjIwYmE4YTE0MTYzLTI2MDMxZTUxLTE2Mzg3MjAtMThiMTg4YmUyY2YxZjYwIiwiJGlkZW50aXR5X2xvZ2luX2lkIjoiMTE2OTM0NTM5NSJ9%22%2C%22history_login_id%22%3A%7B%22name%22%3A%22%24identity_login_id%22%2C%22value%22%3A%221169345395%22%7D%2C%22%24device_id%22%3A%2218b188be2ce11d5-09620ba8a14163-26031e51-1638720-18b188be2cf1f60%22%7D; acw_tc=276077c116969923409976454e308f57c392d1966f936748f70982813451b9; Hm_lvt_38ba284938d5eddca645bb5e02a02006=1696923903,1696992344; Hm_lpvt_38ba284938d5eddca645bb5e02a02006=1696992344; FSSBBIl1UgzbN7NP=5R13WYKVbkm3qqqDhotq0.aJBzrNN267bwmRaUY4yM9RbC8Sm7Z_SPGQhz1srgMogvzspOBpzuEx_omVEzPbCnjOcPfzFmrWC031gS6o3TsDGwvWIMlCbERbAJTvLPrNzIwxTtIpZL0M3PN5IbYB0lLw7CA6h32V5FgJ_Z0vTTjUVgdKukChuTIBGeuka91NYiJR3BxWOkicTpJkwMyG_aHlR9m4W28A5s8lfxR1iHh8fslwGMmvHr1Wfihrn4eR1BD0qecTZ_Pv8fPPczNrqYqHJn4kjlAQfPGsUEhTUqJfq'
    }
    html = requests.get(url,headers = headers).text
    return html

# 解析网页源代码，提取想要的信息
def transform_html(response):
    """
    解析网页源代码，提取想要的信息，并返回信息的dataframe
    response：抓取到的网页源代码
    """
    # 职位名称、薪资、地区、工作经验、学历、职位类别、招聘人数、职位描述、岗位职责、职位福利、公司名称、公司类型、公司规模
    html = etree.HTML(response)
    # 获取工作名称
    job = html.xpath('//span[@class="iteminfo__line1__jobname__name"]/@title')
    # 获取薪资范围
    salary = html.xpath('//p[@class="iteminfo__line2__jobdesc__salary"]/text()')
    for i in range(len(salary)):
        salary[i] = salary[i].strip('\n').strip(' ').rstrip('\n')
    # 获取地区、经验、学历信息
    location,experience,education = ([] for i in range(3))
    require = html.xpath('//ul[@class="iteminfo__line2__jobdesc__demand"]')
    for req in require:
        try:
            loc = req.xpath('.//li[@class="iteminfo__line2__jobdesc__demand__item"]/text()')[0]
            location.append(loc)
        except:
            location.append(np.nan)
        try:
            exp = req.xpath('.//li[@class="iteminfo__line2__jobdesc__demand__item"]/text()')[1]
            experience.append(exp)
        except:
            experience.append(np.nan)
        try:
            edu = req.xpath('.//li[@class="iteminfo__line2__jobdesc__demand__item"]/text()')[2]
            education.append(edu)
        except:
            education.append(np.nan)

    # 获取职位标签
    job_tag = []
    job_tag_lis = html.xpath('//div[@class="iteminfo__line3__welfare"]')
    for tag in job_tag_lis:
        tag_info = tag.xpath('.//div[@class="iteminfo__line3__welfare__item"]/text()')
        tag_info = str(tag_info)
        job_tag.append(tag_info)

    # 获取公司名称
    company_name = html.xpath('//span[@class="iteminfo__line1__compname__name"]/text()')

    # 获取公司类型、公司规模
    company_type = []
    company_size = []
    company_detail = html.xpath('//div[@class="iteminfo__line2__compdesc"]')
    for company in company_detail:
        try:
            com_type = company.xpath('.//span[@class="iteminfo__line2__compdesc__item"]/text()')[0]
            company_type.append(com_type)
        except:
            company_type.append(np.nan)
        try:
            com_size = company.xpath('.//span[@class="iteminfo__line2__compdesc__item"]/text()')[1]
            company_size.append(com_size)
        except:
            company_size.append(np.nan)

    data_lis = [job,salary,location,experience,education,job_tag,company_name,company_type,company_size]
    # 爬取结果合成一个dataframe
    get_data = pd.DataFrame(columns = ['职位名称','薪资范围','地点','工作经验','学历要求','岗位标签','公司名称','公司类型','公司规模'])
    for col,data in zip(get_data.columns,data_lis):
        get_data[col] = data
    # 返回数据的dataframe
    return get_data

# 循环爬取每一页的数据，并且合成为一个dataframe
def concat_data(url_lis):
    # 定义字典储存dataframe
    final_df_dict = {}
    for url,num in zip(url_lis,range(len(url_lis))):
        try:
            print('开始爬取第{}页'.format(num+1))
            # 获取网页源代码
            response = get_html(url)
            # 解析网页源代码并且生成一个dataframe
            final_df = transform_html(response)
            # 将dataframe保存到字典里
            final_df_dict[num] = final_df
            print('第{}页爬取完成'.format(num+1))
            # 爬取完成后程序休眠8秒
            time.sleep(2)
        except:
            print('所有页码都爬取完成！总计爬取{}页'.format(num+1))
    concat_df = pd.concat(list(final_df_dict.values()),ignore_index = True)
    return concat_df

# 爬取结果保存到csv
def save_df(df,city):
    file_name = '{}招聘信息.csv'.format(city)
    path = '成都.csv'.format(file_name)
    df.to_csv(path,encoding = 'utf-8',index = False)
    print('{}保存成功'.format(file_name))
    return

def main():
    city,page = get_city_page()
    url_lis = create_url_lis(city,page)
    concat_df = concat_data(url_lis)
    save_df(concat_df,city)
    return print('所有程序执行完毕')

main()